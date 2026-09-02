/* OpenFOAM v2512 compressibleInterFoam derivative used for Cong B-H3.
   This qualification variant restores the official v2512 thermal equation
   and changes only the passive opening-valve momentum loss. */

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

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Two-phase compressible VOF solver with passive B-H3 valve resistance"
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

        // For this 2D extrusion the reference area is D times extrusion.
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
            waterRhoPhi = alphaPhi1*fvc::interpolate(rho1);
            airRhoPhi = rhoPhi - waterRhoPhi;
            turbulence.correctPhasePhi();

            #include "UEqn.H"
            volScalarField divUp
            (
                "divUp",
                fvc::div(fvc::absolute(phi, U), p)
            );
            #include "TEqn.H"

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
