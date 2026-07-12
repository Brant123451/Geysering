/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) OpenCFD OpenCFD Ltd.
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
    bh1CompressibleInterFoam

Description
    OpenCFD v2512 compressibleInterFoam with a total-sensible-enthalpy
    temperature equation.

    The pressure work is represented by the pressure time derivative instead
    of p*div(U).  This is the standard OpenFOAM total-enthalpy transformation:
    it is energetically equivalent to total internal energy but avoids
    subtracting large atmospheric-pressure work terms in low-Mach open air.
    The two compressible thermodynamic phases and VOF transport equations are
    unchanged. Opening-time sensitivity can additionally activate a passive,
    semi-implicit Forchheimer resistance in the declared valve cell zone.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "CMULES.H"
#include "EulerDdtScheme.H"
#include "localEulerDdtScheme.H"
#include "CrankNicolsonDdtScheme.H"
#include "subCycle.H"
#include "compressibleInterPhaseTransportModel.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "fvcSmooth.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Two-phase compressible VOF solver using the total-enthalpy "
        "temperature equation"
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createTimeControls.H"
    #include "createFields.H"

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
    const word valveResistanceZone
    (
        valveProperties.get<word>("cellZone")
    );
    const scalar valveOpeningDuration =
        valveProperties.get<scalar>("openingDuration");
    const scalar valveMinimumAreaFraction =
        valveProperties.get<scalar>("minimumAreaFraction");
    const scalar valveResistanceLength =
        valveProperties.get<scalar>("resistanceLength");
    label valveResistanceZoneID = -1;

    if (valveActive)
    {
        if (valveModel != "sineSquaredAreaForchheimer")
        {
            FatalErrorInFunction
                << "Unsupported valve model " << valveModel << nl
                << exit(FatalError);
        }
        if (valveOpeningDuration <= SMALL)
        {
            FatalErrorInFunction
                << "openingDuration must be positive for an active valve" << nl
                << exit(FatalError);
        }
        if
        (
            valveMinimumAreaFraction <= 0
         || valveMinimumAreaFraction >= 1
        )
        {
            FatalErrorInFunction
                << "minimumAreaFraction must lie between zero and one" << nl
                << exit(FatalError);
        }
        if (valveResistanceLength <= SMALL)
        {
            FatalErrorInFunction
                << "resistanceLength must be positive" << nl
                << exit(FatalError);
        }

        valveResistanceZoneID =
            mesh.cellZones().findZoneID(valveResistanceZone);
        if (valveResistanceZoneID < 0)
        {
            FatalErrorInFunction
                << "Cannot find valve cellZone " << valveResistanceZone << nl
                << exit(FatalError);
        }

        const labelList& valveCells =
            mesh.cellZones()[valveResistanceZoneID];
        label valveCellCount = valveCells.size();
        scalar valveZoneVolume = 0;
        for (const label celli : valveCells)
        {
            valveZoneVolume += mesh.V()[celli];
        }
        reduce(valveCellCount, sumOp<label>());
        reduce(valveZoneVolume, sumOp<scalar>());

        const scalar pipeArea =
            constant::mathematical::pi*sqr(scalar(0.025));
        const scalar equivalentZoneLength = valveZoneVolume/pipeArea;
        if
        (
            valveCellCount == 0
         || mag(equivalentZoneLength/valveResistanceLength - 1) > 0.2
        )
        {
            FatalErrorInFunction
                << "Valve zone volume " << valveZoneVolume
                << " m3 represents " << equivalentZoneLength
                << " m of the 50 mm pipe, inconsistent with resistanceLength "
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
    volScalarField dpdt
    (
        IOobject
        (
            "dpdt",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(p.dimensions()/dimTime, Zero)
    );
    surfaceScalarField waterRhoPhi
    (
        IOobject
        (
            "waterRhoPhi",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        alphaPhi10*fvc::interpolate(rho1)
    );
    surfaceScalarField airRhoPhi
    (
        IOobject
        (
            "airRhoPhi",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        rhoPhi - waterRhoPhi
    );

    if (!LTS)
    {
        #include "readTimeControls.H"
        #include "CourantNo.H"
        #include "setInitialDeltaT.H"
    }

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readTimeControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "CourantNo.H"
            #include "alphaCourantNo.H"
            #include "setDeltaT.H"
        }

        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (pimple.loop())
        {
            #include "alphaControls.H"
            #include "compressibleAlphaEqnSubCycle.H"

            // Register the transported phase mass fluxes for exact open-boundary
            // water and gas budgets.  Defining the air flux as the residual makes
            // their sum identically equal to the solver's mixture rhoPhi.
            waterRhoPhi = alphaPhi1*fvc::interpolate(rho1);
            airRhoPhi = rhoPhi - waterRhoPhi;

            turbulence.correctPhasePhi();

            #include "UEqn.H"
            #include "TEqn.H"

            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            // Store the pressure derivative for the next enthalpy predictor,
            // as in the standard OpenFOAM compressible PIMPLE solvers.
            dpdt = fvc::ddt(p);

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
