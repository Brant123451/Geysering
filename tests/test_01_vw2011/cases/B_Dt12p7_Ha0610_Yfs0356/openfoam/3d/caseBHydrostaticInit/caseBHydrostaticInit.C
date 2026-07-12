/*---------------------------------------------------------------------------*\
  Case-B discrete hydrostatic pressure initialisation.

  Projects p_rgh onto the same face force used by compressibleInterFlow:

      snGrad(p_rgh) = -ghf*snGrad(rho)

  The conformal closed-valve baffle creates an unvented connected component.
  One pressure reference is therefore retained in every component that has no
  fixed-value p_rgh boundary.
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "constrainPressure.H"
#include "gravityMeshObject.H"
#include "regionSplit.H"
#include "twoPhaseModelThermo.H"

using namespace Foam;

namespace
{

struct residualStats
{
    scalar force;
    vector location;
    scalar algebraic;
};


residualStats measureResidual
(
    const fvMesh& mesh,
    const surfaceScalarField& ghf,
    const volScalarField& gh,
    const volScalarField& rho,
    const volScalarField& p,
    const volScalarField& p_rgh
)
{
    const surfaceScalarField residual
    (
        IOobject
        (
            "caseBHydrostaticResidual",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
       -ghf*fvc::snGrad(rho) - fvc::snGrad(p_rgh)
    );

    scalar maxForce = -GREAT;
    vector maxLocation = vector::zero;
    for (label faceI = 0; faceI < mesh.nInternalFaces(); ++faceI)
    {
        const scalar magnitude = mag(residual[faceI]);
        if (magnitude > maxForce)
        {
            maxForce = magnitude;
            maxLocation = mesh.Cf()[faceI];
        }
    }

    scalar maxAlgebraic = 0;
    forAll(p_rgh, cellI)
    {
        maxAlgebraic = max
        (
            maxAlgebraic,
            mag(p_rgh[cellI] - (p[cellI] - rho[cellI]*gh[cellI]))
        );
    }

    return {maxForce, maxLocation, maxAlgebraic};
}


void reportResidual
(
    const word& stage,
    const label iteration,
    const residualStats& stats
)
{
    Info<< "CASEB_HYDROSTATIC_INIT stage=" << stage
        << " iteration=" << iteration
        << " maxForceResidual(Pa/m)=" << stats.force
        << " location=" << stats.location
        << " maxAlgebraicResidual(Pa)=" << stats.algebraic
        << nl;
}

} // End anonymous namespace


int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Initialise Case-B p and p_rgh with the discrete gravity-force operator"
    );
    argList::noParallel();
    argList::addOption
    (
        "nCorrectors",
        "label",
        "Number of pressure-density hydrostatic corrections (default 5)"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    const label nCorrectors = args.getOrDefault<label>("nCorrectors", 5);
    if (nCorrectors < 1)
    {
        FatalErrorInFunction
            << "nCorrectors must be positive" << exit(FatalError);
    }

    Info<< "Reading field p_rgh" << nl << endl;
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

    Info<< "Reading field U" << nl << endl;
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

    #include "createPhi.H"

    Info<< "Constructing twoPhaseModelThermo" << nl << endl;
    twoPhaseModelThermo mixture(U, phi);
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
            IOobject::NO_WRITE
        ),
        alpha1*rho1 + alpha2*rho2
    );

    const uniformDimensionedVectorField& g =
        meshObjects::gravity::New(runTime);
    dimensionedScalar ghRef("ghRef", g.dimensions()*dimLength, 0);
    volScalarField gh("gh", (g & mesh.C()) - ghRef);
    surfaceScalarField ghf("ghf", (g & mesh.Cf()) - ghRef);

    const surfaceScalarField onef
    (
        IOobject
        (
            "caseBHydrostaticOne",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        mesh,
        dimensionedScalar("one", dimless, 1)
    );

    p_rgh.correctBoundaryConditions();
    rho.correctBoundaryConditions();

    const regionSplit cellRegion(mesh);
    const label nRegions = cellRegion.nRegions();
    labelList firstCell(nRegions, -1);
    boolList hasFixedPressure(nRegions, false);
    forAll(cellRegion, cellI)
    {
        const label regionI = cellRegion[cellI];
        if (firstCell[regionI] < 0)
        {
            firstCell[regionI] = cellI;
        }
    }

    forAll(mesh.boundary(), patchI)
    {
        if (!p_rgh.boundaryField()[patchI].fixesValue())
        {
            continue;
        }

        const labelUList& faceCells = mesh.boundary()[patchI].faceCells();
        forAll(faceCells, faceI)
        {
            hasFixedPressure[cellRegion[faceCells[faceI]]] = true;
        }
    }

    label nReferences = 0;
    forAll(hasFixedPressure, regionI)
    {
        if (!hasFixedPressure[regionI])
        {
            ++nReferences;
        }
    }

    labelList referenceCells(nReferences);
    scalarField referenceValues(nReferences);
    label referenceI = 0;
    forAll(hasFixedPressure, regionI)
    {
        if (!hasFixedPressure[regionI])
        {
            referenceCells[referenceI] = firstCell[regionI];
            referenceValues[referenceI] = p_rgh[firstCell[regionI]];
            ++referenceI;
        }
    }

    Info<< "CASEB_HYDROSTATIC_REGIONS count=" << nRegions
        << " unventedReferences=" << nReferences << nl;
    forAll(referenceCells, refI)
    {
        Info<< "CASEB_HYDROSTATIC_REFERENCE cell=" << referenceCells[refI]
            << " region=" << cellRegion[referenceCells[refI]]
            << " value(Pa)=" << referenceValues[refI]
            << " location=" << mesh.C()[referenceCells[refI]] << nl;
    }

    const residualStats before =
        measureResidual(mesh, ghf, gh, rho, p, p_rgh);
    reportResidual("before", 0, before);

    residualStats after = before;
    for (label corr = 0; corr < nCorrectors; ++corr)
    {
        rho = alpha1*rho1 + alpha2*rho2;
        rho.correctBoundaryConditions();
        p_rgh.correctBoundaryConditions();

        const surfaceScalarField gravityForce
        (
            IOobject
            (
                "caseBGravityForce",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
           -ghf*fvc::snGrad(rho)
        );
        surfaceScalarField forceFlux
        (
            IOobject
            (
                "caseBGravityForceFlux",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            gravityForce*mesh.magSf()
        );

        constrainPressure(p_rgh, U, forceFlux, onef);

        fvScalarMatrix hydrostaticEqn
        (
            fvm::laplacian(onef, p_rgh) == fvc::div(forceFlux)
        );
        hydrostaticEqn.setReferences
        (
            referenceCells,
            referenceValues,
            true
        );
        hydrostaticEqn.solve();

        p = p_rgh + rho*gh;
        p.correctBoundaryConditions();
        mixture.correctThermo();
        mixture.correct();
        rho = alpha1*rho1 + alpha2*rho2;
        rho.correctBoundaryConditions();
        p_rgh.correctBoundaryConditions();

        after = measureResidual(mesh, ghf, gh, rho, p, p_rgh);
        reportResidual("corrector", corr + 1, after);
    }

    p = p_rgh + rho*gh;
    p.correctBoundaryConditions();
    after = measureResidual(mesh, ghf, gh, rho, p, p_rgh);
    reportResidual("final", nCorrectors, after);
    if (!p.write() || !p_rgh.write())
    {
        FatalErrorInFunction
            << "Failed to write hydrostatic p and p_rgh fields"
            << exit(FatalError);
    }

    Info<< "CASEB_HYDROSTATIC_INIT_SUMMARY"
        << " correctors=" << nCorrectors
        << " beforeMaxForceResidual(Pa/m)=" << before.force
        << " afterMaxForceResidual(Pa/m)=" << after.force
        << " afterMaxAlgebraicResidual(Pa)=" << after.algebraic
        << nl;
    Info<< "End" << nl << endl;

    return 0;
}

// ************************************************************************* //
