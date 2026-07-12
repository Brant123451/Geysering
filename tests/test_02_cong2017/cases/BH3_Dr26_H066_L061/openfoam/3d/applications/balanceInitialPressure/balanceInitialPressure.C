#include "fixedFluxPressureFvPatchScalarField.H"
#include "fvCFD.H"
#include "regionSplit.H"
#include "timeSelector.H"
#include "twoPhaseMixtureThermo.H"

using namespace Foam;

int main(int argc, char* argv[])
{
    argList::addNote
    (
        "Balance the t=0 compressibleInterFoam pressure against gravity "
        "and CSF on a closed-valve mesh"
    );
    argList::noFunctionObjects();
    argList::noParallel();
    timeSelector::addOptions_singleTime();

    #include "addRegionOption.H"
    #include "setRootCase.H"
    #include "createTime.H"

    timeSelector::setTimeIfPresent(runTime, args, true);

    if (mag(runTime.value()) > SMALL)
    {
        FatalErrorInFunction
            << "Only t=0 initialisation is permitted; use -time 0"
            << exit(FatalError);
    }

    #include "createNamedMesh.H"

    const label valveUp =
        mesh.boundaryMesh().findPatchID("valve_upstream");
    const label valveDown =
        mesh.boundaryMesh().findPatchID("valve_downstream");

    if
    (
        valveUp < 0
     || valveDown < 0
     || mesh.boundaryMesh()[valveUp].coupled()
     || mesh.boundaryMesh()[valveDown].coupled()
    )
    {
        FatalErrorInFunction
            << "A closed, non-coupled valve baffle is required. "
            << "Balancing an open cyclic valve would erase its physical "
            << "initial pressure jump."
            << exit(FatalError);
    }

    IOdictionary controls
    (
        IOobject
        (
            "balanceInitialPressureDict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const label nOuter =
        controls.getOrDefault<label>("nOuterCorrectors", 40);
    const label nEos =
        controls.getOrDefault<label>("nEosCorrectors", 8);
    const label nNonOrth =
        controls.getOrDefault<label>("nNonOrthogonalCorrectors", 2);
    const scalar pTolerance =
        controls.getOrDefault<scalar>("pTolerance", 1e-5);
    const scalar rhoTolerance =
        controls.getOrDefault<scalar>("rhoTolerance", 1e-10);
    const scalar relaxation =
        controls.getOrDefault<scalar>("relaxation", 1);
    const scalar zeroUTolerance =
        controls.getOrDefault<scalar>("zeroVelocityTolerance", 1e-12);

    if
    (
        nOuter < 1
     || nEos < 1
     || nNonOrth < 0
     || pTolerance <= 0
     || rhoTolerance <= 0
     || relaxation <= 0
     || relaxation > 1
     || zeroUTolerance < 0
    )
    {
        FatalErrorInFunction
            << "Invalid controls in balanceInitialPressureDict"
            << exit(FatalError);
    }

    volScalarField p_rgh
    (
        IOobject
        (
            "p_rgh",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    volVectorField U
    (
        IOobject
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        mesh
    );

    if (gMax(mag(U.primitiveField())) > zeroUTolerance)
    {
        FatalErrorInFunction
            << "Static initialisation requires U=0 within "
            << zeroUTolerance
            << exit(FatalError);
    }

    surfaceScalarField phi("phi", fvc::flux(U));
    twoPhaseMixtureThermo mixture(U, phi);

    volScalarField& p = mixture.p();
    volScalarField& alpha1 = mixture.alpha1();
    volScalarField& alpha2 = mixture.alpha2();
    const volScalarField& rho1 = mixture.thermo1().rho();
    const volScalarField& rho2 = mixture.thermo2().rho();

    volScalarField rho
    (
        IOobject
        (
            "rho",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        alpha1*rho1 + alpha2*rho2
    );

    const dimensionedScalar pMin("pMin", dimPressure, mixture);

    #include "readGravitationalAcceleration.H"
    #include "readhRef.H"
    #include "gh.H"

    const auto requireFixedPressure =
        [&](const word& patchName)
        {
            const label patchi =
                mesh.boundaryMesh().findPatchID(patchName);

            if
            (
                patchi < 0
             || !p_rgh.boundaryField()[patchi].fixesValue()
            )
            {
                FatalErrorInFunction
                    << patchName << " must fix p_rgh"
                    << exit(FatalError);
            }
        };

    requireFixedPressure("inlet");
    requireFixedPressure("atmosphere");

    if
    (
        !isA<fixedFluxPressureFvPatchScalarField>
        (
            p_rgh.boundaryField()[valveUp]
        )
     || !isA<fixedFluxPressureFvPatchScalarField>
        (
            p_rgh.boundaryField()[valveDown]
        )
    )
    {
        FatalErrorInFunction
            << "Closed valve patches must use fixedFluxPressure"
            << exit(FatalError);
    }

    regionSplit regions(mesh);
    const label nRegions = regions.nRegions();
    boolList hasFixedPressure(nRegions, false);

    forAll(p_rgh.boundaryField(), patchi)
    {
        if (p_rgh.boundaryField()[patchi].fixesValue())
        {
            const labelUList& faceCells =
                mesh.boundary()[patchi].faceCells();

            forAll(faceCells, facei)
            {
                hasFixedPressure[regions[faceCells[facei]]] = true;
            }
        }
    }

    labelList refCellByRegion(nRegions, -1);

    forAll(regions, celli)
    {
        const label regioni = regions[celli];

        if
        (
            !hasFixedPressure[regioni]
         && refCellByRegion[regioni] < 0
        )
        {
            refCellByRegion[regioni] = celli;
        }
    }

    label nReferences = 0;
    forAll(refCellByRegion, regioni)
    {
        if (refCellByRegion[regioni] >= 0)
        {
            ++nReferences;
        }
    }

    labelList referenceCells(nReferences);
    scalarField referenceValues(nReferences);
    label refi = 0;

    forAll(refCellByRegion, regioni)
    {
        const label celli = refCellByRegion[regioni];

        if (celli >= 0)
        {
            referenceCells[refi] = celli;
            referenceValues[refi] = p_rgh[celli];

            Info<< "Anchoring all-Neumann region " << regioni
                << " at cell " << celli
                << ", C=" << mesh.C()[celli]
                << ", p_rgh=" << p_rgh[celli] << nl;

            ++refi;
        }
    }

    const auto preparePressureBoundaries =
        [&](const surfaceScalarField& force)
        {
            volScalarField::Boundary& pBf =
                p_rgh.boundaryFieldRef();

            forAll(pBf, patchi)
            {
                pBf[patchi].setUpdated(false);
            }

            setSnGrad<fixedFluxPressureFvPatchScalarField>
            (
                pBf,
                force.boundaryField()
            );
        };

    const auto correctEos =
        [&]()
        {
            scalar rhoChange = GREAT;

            for (label corr = 0; corr < nEos; ++corr)
            {
                const scalarField rhoOld(rho.primitiveField());

                p = max(p_rgh + rho*gh, pMin);
                mixture.correctThermo();
                rho = alpha1*rho1 + alpha2*rho2;

                rhoChange = gMax
                (
                    mag(rho.primitiveField() - rhoOld)
                );

                if (rhoChange <= rhoTolerance)
                {
                    break;
                }
            }

            p = max(p_rgh + rho*gh, pMin);
            return rhoChange;
        };

    bool converged = false;
    scalar lastFaceResidual = GREAT;
    scalar lastReconstructedResidual = GREAT;

    for (label outer = 0; outer < nOuter; ++outer)
    {
        const scalarField pRghOld(p_rgh.primitiveField());
        const scalarField rhoOld(rho.primitiveField());

        correctEos();
        mixture.correct();

        surfaceScalarField force
        (
            "balanceForce",
            mixture.surfaceTensionForce()
          - ghf*fvc::snGrad(rho)
        );

        for (label nonOrth = 0; nonOrth <= nNonOrth; ++nonOrth)
        {
            preparePressureBoundaries(force);

            fvScalarMatrix pEqn
            (
                fvm::laplacian(p_rgh)
             ==
                fvc::div(force*mesh.magSf())
            );

            if (referenceCells.size())
            {
                pEqn.setReferences
                (
                    referenceCells,
                    referenceValues,
                    true
                );
            }

            pEqn.solve
            (
                p_rgh.select(nonOrth == nNonOrth)
            );
        }

        if (relaxation < 1)
        {
            p_rgh.primitiveFieldRef() =
                pRghOld
              + relaxation
               *(p_rgh.primitiveField() - pRghOld);
        }

        preparePressureBoundaries(force);
        p_rgh.correctBoundaryConditions();

        const scalar eosChange = correctEos();
        mixture.correct();

        surfaceScalarField finalForce
        (
            "finalBalanceForce",
            mixture.surfaceTensionForce()
          - ghf*fvc::snGrad(rho)
        );

        preparePressureBoundaries(finalForce);
        p_rgh.correctBoundaryConditions();

        surfaceScalarField forceResidual
        (
            "initialForceResidual",
            finalForce - fvc::snGrad(p_rgh)
        );
        tmp<volVectorField> reconstructed =
            fvc::reconstruct(forceResidual*mesh.magSf());

        const scalar dp = gMax
        (
            mag(p_rgh.primitiveField() - pRghOld)
        );
        const scalar drho = gMax
        (
            mag(rho.primitiveField() - rhoOld)
        );
        const scalar maxForce =
            gMax(mag(finalForce.primitiveField()));

        lastFaceResidual =
            gMax(mag(forceResidual.primitiveField()));
        lastReconstructedResidual =
            gMax(mag(reconstructed().primitiveField()));

        Info<< "outer=" << outer + 1
            << " maxDeltaP_rgh=" << dp
            << " maxDeltaRho=" << drho
            << " eosDeltaRho=" << eosChange
            << " maxFaceResidual=" << lastFaceResidual
            << " relativeFaceResidual="
            << lastFaceResidual/max(maxForce, VSMALL)
            << " maxReconstructedResidual="
            << lastReconstructedResidual << nl;

        if
        (
            dp <= pTolerance
         && drho <= rhoTolerance
         && eosChange <= rhoTolerance
        )
        {
            converged = true;
            break;
        }
    }

    if (!converged)
    {
        FatalErrorInFunction
            << "EOS/pressure fixed-point iteration did not converge; "
            << "fields were not written"
            << exit(FatalError);
    }

    if (!(p.write() && p_rgh.write() && rho.write()))
    {
        FatalErrorInFunction
            << "Failed writing p, p_rgh, or rho"
            << exit(FatalError);
    }

    Info<< "Final face residual: " << lastFaceResidual << nl
        << "Final reconstructed residual: "
        << lastReconstructedResidual << nl
        << "alpha.water and U were not modified." << nl
        << "End" << endl;

    return 0;
}
