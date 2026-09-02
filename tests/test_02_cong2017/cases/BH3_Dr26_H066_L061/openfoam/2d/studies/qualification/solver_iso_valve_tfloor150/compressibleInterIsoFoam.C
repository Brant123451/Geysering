/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2020 OpenCFD Ltd.
    Copyright (C) 2020 Johan Roenby
    Copyright (C) 2020 DLR
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    compressibleInterFlow

Description
    Solver derived from interFoam for two compressible, immiscible
    fluids using the isoAdvector phase-fraction based interface capturing
    approach, with optional mesh motion and mesh topology changes including
    adaptive re-meshing.

    Reference:
    \verbatim
        Roenby, J., Bredmose, H. and Jasak, H. (2016).
        A computational method for sharp interface advection
        Royal Society Open Science, 3
        doi 10.1098/rsos.160405
    \endverbatim

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "CMULES.H"
#include "EulerDdtScheme.H"
#include "localEulerDdtScheme.H"
#include "CrankNicolsonDdtScheme.H"
#include "subCycle.H"
#include "compressibleInterPhaseTransportModel.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "CorrectPhi.H"
#include "fvcSmooth.H"
#include "dynamicRefineFvMesh.H"
#include "isoAdvection.H"
#include "twoPhaseMixtureThermo.H"
#include "mathematicalConstants.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "B-H3 compressible isoAdvector solver with passive valve resistance."
    );

    #include "postProcess.H"

    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "initContinuityErrs.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "createUf.H"
    #include "CourantNo.H"
    #include "setInitialDeltaT.H"

    IOdictionary valveProperties
    (
        IOobject
        (
            "valveProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    const Switch valveActive(valveProperties.get<Switch>("active"));
    const word valveModel(valveProperties.get<word>("model"));
    const word valveResistanceZone(valveProperties.get<word>("cellZone"));
    const scalar valveOpeningDuration =
        valveProperties.get<scalar>("openingDuration");
    const scalar valveMinimumAreaFraction =
        valveProperties.get<scalar>("minimumAreaFraction");
    const scalar valveResistanceLength =
        valveProperties.get<scalar>("resistanceLength");
    const scalar valveReferenceFlowArea =
        valveProperties.get<scalar>("referenceFlowArea");
    label valveResistanceZoneID = -1;

    if (valveActive)
    {
        if (valveModel != "sineSquaredAreaForchheimer")
        {
            FatalErrorInFunction
                << "Unsupported valve model " << valveModel << nl
                << exit(FatalError);
        }
        if
        (
            valveOpeningDuration <= SMALL
         || valveResistanceLength <= SMALL
         || valveReferenceFlowArea <= SMALL
         || valveMinimumAreaFraction <= 0
         || valveMinimumAreaFraction >= 1
        )
        {
            FatalErrorInFunction
                << "Invalid valve duration, area, length, or minimum opening"
                << nl << exit(FatalError);
        }

        valveResistanceZoneID =
            mesh.cellZones().findZoneID(valveResistanceZone);
        if (valveResistanceZoneID < 0)
        {
            FatalErrorInFunction
                << "Cannot find valve cellZone " << valveResistanceZone << nl
                << exit(FatalError);
        }

        const labelList& valveCells = mesh.cellZones()[valveResistanceZoneID];
        label valveCellCount = valveCells.size();
        scalar valveZoneVolume = 0;
        for (const label celli : valveCells)
        {
            valveZoneVolume += mesh.V()[celli];
        }
        reduce(valveCellCount, sumOp<label>());
        reduce(valveZoneVolume, sumOp<scalar>());

        const scalar equivalentZoneLength =
            valveZoneVolume/valveReferenceFlowArea;
        if
        (
            valveCellCount == 0
         || mag(equivalentZoneLength/valveResistanceLength - 1) > 0.2
        )
        {
            FatalErrorInFunction
                << "Valve zone volume " << valveZoneVolume
                << " represents " << equivalentZoneLength
                << " m, inconsistent with resistanceLength "
                << valveResistanceLength << " m" << nl
                << exit(FatalError);
        }

        Info<< "Equivalent valve resistance active: duration="
            << valveOpeningDuration << " s, zone=" << valveResistanceZone
            << ", global cells=" << valveCellCount
            << ", equivalent length=" << equivalentZoneLength << " m" << nl;
    }

    volScalarField& p = mixture.p();
    volScalarField& T = mixture.T();
    const volScalarField& psi1 = mixture.thermo1().psi();
    const volScalarField& psi2 = mixture.thermo2().psi();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readDyMControls.H"

        // Store divU and divUp from the previous mesh so that it can be mapped
        // and used in correctPhi to ensure the corrected phi has the
        // same divergence
        volScalarField divU("divU0", fvc::div(fvc::absolute(phi, U)));

        #include "CourantNo.H"
        #include "alphaCourantNo.H"
        #include "setDeltaT.H"


        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            if (pimple.firstIter() || moveMeshOuterCorrectors)
            {
                scalar timeBeforeMeshUpdate = runTime.elapsedCpuTime();

                if (isA<dynamicRefineFvMesh>(mesh))
                {
                    advector.surf().reconstruct();
                }

                mesh.update();

                if (mesh.changing())
                {
                    gh = (g & mesh.C()) - ghRef;
                    ghf = (g & mesh.Cf()) - ghRef;

                    if (isA<dynamicRefineFvMesh>(mesh))
                    {
                        advector.surf().mapAlphaField();
                        alpha2 = 1.0 - alpha1;
                        alpha2.correctBoundaryConditions();
                        rho == alpha1*rho1 + alpha2*rho2;
                        rho.correctBoundaryConditions();
                        rho.oldTime() = rho;
                        alpha2.oldTime() = alpha2;
                    }

                    MRF.update();

                    Info<< "Execution time for mesh.update() = "
                        << runTime.elapsedCpuTime() - timeBeforeMeshUpdate
                        << " s" << endl;

                }

                if ((mesh.changing() && correctPhi))
                {
                    // Calculate absolute flux from the mapped surface velocity
                    phi = mesh.Sf() & Uf;

                    #include "correctPhi.H"

                    // Make the fluxes relative to the mesh motion
                    fvc::makeRelative(phi, U);

                    mixture.correct();
                }

                if (mesh.changing() && checkMeshCourantNo)
                {
                    #include "meshCourantNo.H"
                }
            }

            #include "alphaControls.H"
            #include "compressibleAlphaEqnSubCycle.H"

            turbulence.correctPhasePhi();

            #include "UEqn.H"
            volScalarField divUp("divUp", fvc::div(fvc::absolute(phi, U), p));
            #include "TEqn.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                turbulence.correct();
            }
        }

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
