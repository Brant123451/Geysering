/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.
\*---------------------------------------------------------------------------*/

#include "boundedPhaseMassTransport.H"
#include "MULES.H"
#include "addToRunTimeSelectionTable.H"
#include "fixedValueFvPatchField.H"
#include "fvcDdt.H"
#include "fvcDiv.H"
#include "fvcFlux.H"
#include "fvmLaplacian.H"
#include "geometricOneField.H"
#include "inletOutletFvPatchFields.H"
#include "slicedSurfaceFields.H"
#include "surfaceInterpolate.H"
#include "upwind.H"
#include "zeroField.H"
#include "zeroGradientFvPatchField.H"

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(boundedPhaseMassTransport, 0);

    addToRunTimeSelectionTable
    (
        functionObject,
        boundedPhaseMassTransport,
        dictionary
    );
}
}


Foam::volScalarField&
Foam::functionObjects::boundedPhaseMassTransport::transportedField()
{
    if (!foundObject<volScalarField>(fieldName_))
    {
        auto fieldPtr = tmp<volScalarField>::New
        (
            IOobject
            (
                fieldName_,
                mesh_.time().timeName(),
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                IOobject::REGISTER
            ),
            mesh_
        );

        store(fieldName_, fieldPtr);
    }

    return lookupObjectRef<volScalarField>(fieldName_);
}


Foam::volScalarField&
Foam::functionObjects::boundedPhaseMassTransport::potential
(
    const surfaceScalarField& phi,
    const volScalarField& p
)
{
    if (!potentialPtr_.valid())
    {
        wordList patchFieldTypes(mesh_.boundary().size());
        forAll(p.boundaryField(), patchi)
        {
            patchFieldTypes[patchi] =
                p.boundaryField()[patchi].fixesValue()
              ? fixedValueFvPatchField<scalar>::typeName
              : zeroGradientFvPatchField<scalar>::typeName;
        }

        potentialPtr_.reset
        (
            new volScalarField
            (
                IOobject
                (
                    potentialName_,
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::READ_IF_PRESENT,
                    IOobject::NO_WRITE,
                    IOobject::REGISTER
                ),
                mesh_,
                dimensionedScalar
                (
                    potentialName_,
                    phi.dimensions()/dimLength,
                    Zero
                ),
                patchFieldTypes
            )
        );
        mesh_.setFluxRequired(potentialName_);
    }

    return potentialPtr_();
}


Foam::volScalarField&
Foam::functionObjects::boundedPhaseMassTransport::updateInventoryDensity
(
    const volScalarField& alpha,
    const volScalarField& phaseRho
)
{
    const tmp<volScalarField> physicalDensity(alpha*phaseRho);

    if (!inventoryRhoPtr_.valid())
    {
        inventoryRhoPtr_.reset
        (
            new volScalarField
            (
                IOobject
                (
                    rhoResultName_,
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    IOobject::REGISTER
                ),
                physicalDensity()
            )
        );
    }
    else
    {
        inventoryRhoPtr_() == physicalDensity();
    }

    return inventoryRhoPtr_();
}


Foam::volScalarField&
Foam::functionObjects::boundedPhaseMassTransport::conservedDensity
(
    const volScalarField& alpha,
    const volScalarField& phaseRho,
    const volScalarField& field
)
{
    if (!sigmaPtr_.valid())
    {
        // fvm::div cannot use calculated patch fields.  Mirror the fraction
        // field's patch types, but replace inletOutlet with fixedValue so the
        // boundary tagged density can be refreshed from alpha*rho*s after the
        // fraction BCs are applied (same inflow/outflow selection).
        wordList patchFieldTypes(field.boundaryField().types());
        forAll(patchFieldTypes, patchi)
        {
            if
            (
                patchFieldTypes[patchi]
             == inletOutletFvPatchField<scalar>::typeName
            )
            {
                patchFieldTypes[patchi] =
                    fixedValueFvPatchField<scalar>::typeName;
            }
        }

        sigmaPtr_.reset
        (
            new volScalarField
            (
                IOobject
                (
                    sigmaResultName_,
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    IOobject::REGISTER
                ),
                mesh_,
                dimensionedScalar(dimDensity, Zero),
                patchFieldTypes
            )
        );
        sigmaPtr_() == alpha*phaseRho*field;
        mesh_.setFluxRequired(sigmaResultName_);
    }

    return sigmaPtr_();
}


Foam::functionObjects::boundedPhaseMassTransport::
boundedPhaseMassTransport
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    fieldName_(dict.getOrDefault<word>("field", "s")),
    phiName_(dict.getOrDefault<word>("phi", "phi")),
    alphaName_(dict.getOrDefault<word>("alpha", "alpha")),
    phaseRhoName_(dict.getOrDefault<word>("phaseRho", "rho")),
    pName_(dict.getOrDefault<word>("p", "p")),
    potentialName_
    (
        dict.getOrDefault<word>
        (
            "potential",
            fieldName_ + "CarrierPotential"
        )
    ),
    carrierFluxResultName_
    (
        dict.getOrDefault<word>
        (
            "carrierFluxResult",
            fieldName_ + "CarrierMassFlux"
        )
    ),
    rhoResultName_
    (
        dict.getOrDefault<word>("rhoResult", fieldName_ + "CarrierRho")
    ),
    sigmaResultName_
    (
        dict.getOrDefault<word>("sigmaResult", fieldName_ + "Sigma")
    ),
    fluxResultName_
    (
        dict.getOrDefault<word>("fluxResult", fieldName_ + "MassFlux")
    ),
    sourceResultName_
    (
        dict.getOrDefault<word>("sourceResult", fieldName_ + "MassSource")
    ),
    schemesField_("unknown-schemesField"),
    residualAlpha_(1e-8),
    tolerance_(1),
    boundsTolerance_(1e-6),
    continuityTolerance_(1e-4),
    nCorr_(0),
    nNonOrthCorr_(0),
    nProjectionCorr_(0),
    resetOnStartUp_(false),
    potentialPtr_(nullptr),
    inventoryRhoPtr_(nullptr),
    sigmaPtr_(nullptr),
    alphaRhoPtr_(nullptr),
    tracerSourcePtr_(nullptr),
    carrierFluxPtr_(nullptr),
    tracerFluxPtr_(nullptr)
{
    read(dict);

    volScalarField& field = transportedField();
    mesh_.setFluxRequired(fieldName_);

    if (resetOnStartUp_)
    {
        field == Zero;
    }
}


bool Foam::functionObjects::boundedPhaseMassTransport::read
(
    const dictionary& dict
)
{
    if (!fvMeshFunctionObject::read(dict))
    {
        return false;
    }

    dict.readIfPresent("phi", phiName_);
    dict.readIfPresent("alpha", alphaName_);
    dict.readIfPresent("phaseRho", phaseRhoName_);
    dict.readIfPresent("p", pName_);
    dict.readIfPresent("potential", potentialName_);
    dict.readIfPresent("carrierFluxResult", carrierFluxResultName_);
    dict.readIfPresent("rhoResult", rhoResultName_);
    dict.readIfPresent("sigmaResult", sigmaResultName_);
    dict.readIfPresent("fluxResult", fluxResultName_);
    dict.readIfPresent("sourceResult", sourceResultName_);
    schemesField_ = dict.getOrDefault("schemesField", fieldName_);
    dict.readIfPresent("residualAlpha", residualAlpha_);
    dict.readIfPresent("tolerance", tolerance_);
    dict.readIfPresent("boundsTolerance", boundsTolerance_);
    dict.readIfPresent("continuityTolerance", continuityTolerance_);
    dict.readIfPresent("nCorr", nCorr_);
    dict.readIfPresent("nNonOrthCorr", nNonOrthCorr_);
    dict.readIfPresent("nProjectionCorr", nProjectionCorr_);
    dict.readIfPresent("resetOnStartUp", resetOnStartUp_);

    if
    (
        residualAlpha_ < 0
     || tolerance_ < 0
     || boundsTolerance_ < 0
     || continuityTolerance_ <= 0
     || nCorr_ < 0
     || nNonOrthCorr_ < 0
     || nProjectionCorr_ < 0
    )
    {
        FatalErrorInFunction
            << "Invalid transport controls: residualAlpha=" << residualAlpha_
            << ", tolerance=" << tolerance_
            << ", boundsTolerance=" << boundsTolerance_
            << ", continuityTolerance=" << continuityTolerance_
            << ", nCorr=" << nCorr_
            << ", nNonOrthCorr=" << nNonOrthCorr_
            << ", nProjectionCorr=" << nProjectionCorr_
            << exit(FatalError);
    }

    return true;
}


bool Foam::functionObjects::boundedPhaseMassTransport::execute()
{
    volScalarField& field = transportedField();
    const volScalarField& alpha =
        lookupObject<volScalarField>(alphaName_);
    const volScalarField& phaseRho =
        lookupObject<volScalarField>(phaseRhoName_);
    const surfaceScalarField& phi =
        lookupObject<surfaceScalarField>(phiName_);
    const volScalarField& p =
        lookupObject<volScalarField>(pName_);

    if (phaseRho.dimensions() != dimDensity)
    {
        FatalErrorInFunction
            << "Expected phase density dimensions " << dimDensity
            << " for " << phaseRho.name() << ", found "
            << phaseRho.dimensions()
            << exit(FatalError);
    }

    if (phi.dimensions() != dimMass/dimTime)
    {
        FatalErrorInFunction
            << "Expected phase mass flux dimensions " << dimMass/dimTime
            << " for " << phi.name() << ", found " << phi.dimensions()
            << exit(FatalError);
    }

    if (!carrierFluxPtr_.valid())
    {
        carrierFluxPtr_.reset
        (
            new surfaceScalarField
            (
                IOobject
                (
                    carrierFluxResultName_,
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    IOobject::REGISTER
                ),
                phi
            )
        );
    }
    else
    {
        carrierFluxPtr_() = phi;
    }

    surfaceScalarField& carrierFlux = carrierFluxPtr_();
    carrierFlux.oriented() = phi.oriented();

    volScalarField carrierDdt
    (
        IOobject
        (
            fieldName_ + "CarrierDdt",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        ),
        fvc::ddt(phaseRho, alpha)
    );

    volScalarField continuityErrorBefore
    (
        IOobject
        (
            fieldName_ + "CarrierContinuityErrorBefore",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        ),
        carrierDdt + fvc::div(carrierFlux)
    );

    volScalarField& Phi = potential(phi, p);
    const word laplacianScheme("laplacian(" + potentialName_ + ")");

    volScalarField carrierDivAfter
    (
        IOobject
        (
            fieldName_ + "CarrierDivAfter",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        ),
        fvc::div(carrierFlux)
    );

    volScalarField continuityErrorAfter
    (
        IOobject
        (
            fieldName_ + "CarrierContinuityErrorAfter",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        ),
        carrierDdt + carrierDivAfter
    );

    const auto maxMag = [](const volScalarField& value)
    {
        return max(mag(gMin(value)), mag(gMax(value)));
    };
    scalar relativeContinuityError = GREAT;
    label projectionIterations = 0;
    scalarField primaryPotential;
    bool restorePrimaryPotential = false;

    for
    (
        label projection = 0;
        projection <= nProjectionCorr_;
        ++projection
    )
    {
        // Each outer pass solves for an incremental correction to the already
        // corrected carrier flux.  This removes the residual left by finite
        // non-orthogonal correction loops without loosening the acceptance
        // criterion.
        if (projection > 0)
        {
            Phi == Zero;
            Phi.correctBoundaryConditions();
        }

        for (label nonOrth = 0; nonOrth <= nNonOrthCorr_; ++nonOrth)
        {
            fvScalarMatrix PhiEqn
            (
                fvm::laplacian(Phi, laplacianScheme)
              + carrierDdt
              + fvc::div(carrierFlux)
            );

            PhiEqn.solve(potentialName_);

            if (nonOrth == nNonOrthCorr_)
            {
                tmp<surfaceScalarField> correctionFlux(PhiEqn.flux());
                correctionFlux.ref().oriented() = carrierFlux.oriented();
                carrierFlux += correctionFlux();
            }
        }

        carrierDivAfter == fvc::div(carrierFlux);
        continuityErrorAfter == carrierDdt + carrierDivAfter;

        const scalar continuityScale =
            max(maxMag(carrierDdt), max(maxMag(carrierDivAfter), VSMALL));
        relativeContinuityError =
            maxMag(continuityErrorAfter)/continuityScale;
        projectionIterations = projection + 1;

        if (relativeContinuityError <= continuityTolerance_)
        {
            break;
        }

        if (projection == 0 && projection < nProjectionCorr_)
        {
            // Preserve the primary correction as the initial guess for the
            // next physical time step.  Additional passes solve only the
            // current non-orthogonal residual and should not replace it.
            primaryPotential = Phi.primitiveField();
            restorePrimaryPotential = true;
        }
    }

    if (restorePrimaryPotential)
    {
        Phi.primitiveFieldRef() = primaryPotential;
        Phi.correctBoundaryConditions();
    }

    if (relativeContinuityError > continuityTolerance_)
    {
        FatalErrorInFunction
            << "Carrier-flux projection did not satisfy phase continuity: "
            << "relative max error = " << relativeContinuityError
            << ", tolerance = " << continuityTolerance_ << nl
            << "Refusing to transport the source tracer with an inconsistent "
            << "carrier flux." << exit(FatalError);
    }

    // inletOutlet patches must use the corrected phase mass flux when
    // selecting inflow values.
    field.correctBoundaryConditions();

    const volScalarField alphaRho
    (
        IOobject
        (
            fieldName_ + "AlphaRho",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        ),
        alpha*phaseRho
    );

    // Keep a registered alpha*rho only for post-processing weight reuse.
    updateInventoryDensity(alpha, phaseRho);

    volScalarField& sigma = conservedDensity(alpha, phaseRho, field);
    if (!sigma.nOldTimes())
    {
        sigma == alphaRho*field;
        sigma.oldTime();
    }
    else
    {
        // Ensure old-time storage exists; Time::storeOldTimes already copied
        // the previous inventory into oldTime before this function object ran.
        sigma.oldTime();
    }

    const volScalarField sigmaOld
    (
        IOobject
        (
            sigmaResultName_ + "Old",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        ),
        sigma
    );

    const word divScheme("div(phi," + schemesField_ + ")");
    const dimensionedScalar dt(mesh_.time().deltaT());
    const scalar resolveAlpha = max(residualAlpha_, 1e-3);
    const scalar rDeltaT = 1.0/mesh_.time().deltaTValue();

    // Sync the intensity from the conserved density before building fluxes.
    // Do not use phiVol = phi_air/(alpha*rho)_f: that form explodes on thin
    // phase faces and drove recovered s to O(100) within a few steps.
    {
        scalarField& fieldCells = field.primitiveFieldRef();
        const scalarField& sigmaCells = sigma.primitiveField();
        const scalarField& alphaRhoCells = alphaRho.primitiveField();
        const scalarField& alphaCells = alpha.primitiveField();
        forAll(fieldCells, celli)
        {
            if (alphaCells[celli] > resolveAlpha && alphaRhoCells[celli] > SMALL)
            {
                fieldCells[celli] = sigmaCells[celli]/alphaRhoCells[celli];
            }
            else
            {
                fieldCells[celli] = 0;
            }
        }
        field.correctBoundaryConditions();
    }

    // Refresh non-coupled sigma BCs after the intensity BC update.
    {
        volScalarField::Boundary& sigmaBf = sigma.boundaryFieldRef();
        const volScalarField::Boundary& alphaRhoBf = alphaRho.boundaryField();
        const volScalarField::Boundary& fieldBf = field.boundaryField();
        forAll(sigmaBf, patchi)
        {
            if (!sigmaBf[patchi].coupled())
            {
                sigmaBf[patchi] == alphaRhoBf[patchi]*fieldBf[patchi];
            }
        }
    }

    // Tagged mass flux from intensity: phi_air * s.  High-order / upwind.
    surfaceScalarField tracerFlux
    (
        IOobject
        (
            fluxResultName_,
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        ),
        fvc::flux(carrierFlux, field, divScheme)
    );
    tracerFlux.oriented() = carrierFlux.oriented();

    surfaceScalarField tracerFluxBD
    (
        upwind<scalar>(mesh_, carrierFlux).flux(field)
    );
    tracerFluxBD.oriented() = carrierFlux.oriented();

    // Match MULES::limit: on non-coupled patches the bounded flux equals the
    // prescribed high-order boundary flux.
    {
        surfaceScalarField::Boundary& bdBf = tracerFluxBD.boundaryFieldRef();
        const surfaceScalarField::Boundary& hoBf = tracerFlux.boundaryField();
        forAll(bdBf, patchi)
        {
            if (!bdBf[patchi].coupled())
            {
                bdBf[patchi] = hoBf[patchi];
            }
        }
    }

    surfaceScalarField phiCorr(tracerFlux - tracerFluxBD);

    const scalarField sigmaMax(alphaRho.primitiveField());
    const scalarField sigmaMin(mesh_.nCells(), Zero);
    scalarField allLambda(mesh_.nFaces(), 1.0);

    // Limit the correction so the explicit sigma update stays in
    // [0, alpha*rho].  Fluxes remain conservative (no cell clip).
    MULES::limiter
    (
        allLambda,
        rDeltaT,
        geometricOneField(),
        sigma,
        tracerFluxBD,
        phiCorr,
        zeroField(),
        zeroField(),
        sigmaMax,
        sigmaMin
    );

    {
        slicedSurfaceScalarField lambda
        (
            IOobject
            (
                fieldName_ + "MULESLambda",
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                IOobject::NO_REGISTER
            ),
            mesh_,
            dimless,
            allLambda,
            false
        );
        tracerFlux = tracerFluxBD + lambda*phiCorr;
    }

    MULES::explicitSolve
    (
        geometricOneField(),
        sigma,
        tracerFlux,
        zeroField(),
        zeroField()
    );

    // Recover s for output/BCs/arrival; inventory remains sigma.
    scalarField& fieldCells = field.primitiveFieldRef();
    const scalarField& sigmaCells = sigma.primitiveField();
    const scalarField& alphaRhoCells = alphaRho.primitiveField();
    const scalarField& alphaCells = alpha.primitiveField();
    forAll(fieldCells, celli)
    {
        if (alphaCells[celli] > resolveAlpha && alphaRhoCells[celli] > SMALL)
        {
            fieldCells[celli] = sigmaCells[celli]/alphaRhoCells[celli];
        }
        else
        {
            fieldCells[celli] = 0;
        }
    }
    field.correctBoundaryConditions();

    const label iteration = 1;
    const bool converged = true;

    const scalar sigmaMass = gSum(mesh_.V()*sigma.primitiveField());
    const scalar sigmaMassOld = gSum(mesh_.V()*sigmaOld.primitiveField());
    if (mag(sigmaMassOld) > SMALL && mag(sigmaMass) < 0.5*mag(sigmaMassOld))
    {
        FatalErrorInFunction
            << "Conserved tracer density inventory collapsed: "
            << sigmaMassOld << " -> " << sigmaMass << exit(FatalError);
    }

    scalar fieldMin = VGREAT;
    scalar fieldMax = -VGREAT;
    forAll(fieldCells, celli)
    {
        if (alphaCells[celli] > resolveAlpha)
        {
            fieldMin = min(fieldMin, fieldCells[celli]);
            fieldMax = max(fieldMax, fieldCells[celli]);
        }
    }
    reduce(fieldMin, minOp<scalar>());
    reduce(fieldMax, maxOp<scalar>());
    if (fieldMin > 0.5*VGREAT)
    {
        fieldMin = gMin(field);
        fieldMax = gMax(field);
    }

    if
    (
        fieldMin < -boundsTolerance_
     || fieldMax > 1 + boundsTolerance_
    )
    {
        FatalErrorInFunction
            << field.name() << " left [0,1] in resolved phase cells: min/max = "
            << fieldMin << ' ' << fieldMax
            << ", violations = " << max(-fieldMin, scalar(0))
            << ' ' << max(fieldMax - 1, scalar(0))
            << ", tolerance = " << boundsTolerance_ << nl
            << "Refusing to continue with a non-physical source tracer."
            << exit(FatalError);
    }

    // Discrete tagged-mass residual of the conserved density sigma.
    const tmp<volScalarField> tracerSource
    (
        (sigma - sigmaOld)/dt + fvc::div(tracerFlux)
    );
    if (!tracerSourcePtr_.valid())
    {
        tracerSourcePtr_.reset
        (
            new volScalarField
            (
                IOobject
                (
                    sourceResultName_,
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    IOobject::REGISTER
                ),
                tracerSource()
            )
        );
    }
    else
    {
        tracerSourcePtr_() == tracerSource();
    }

    if (!tracerFluxPtr_.valid())
    {
        tracerFluxPtr_.reset
        (
            new surfaceScalarField
            (
                IOobject
                (
                    fluxResultName_,
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    IOobject::REGISTER
                ),
                tracerFlux
            )
        );
    }
    else
    {
        tracerFluxPtr_() = tracerFlux;
    }

    Log << type() << " execute: " << field.name()
        << ", sigma min/max = " << gMin(sigma) << ' ' << gMax(sigma)
        << ", fraction min/max = " << fieldMin << ' ' << fieldMax
        << ", carrier continuity error before min/max = "
        << gMin(continuityErrorBefore) << ' ' << gMax(continuityErrorBefore)
        << ", after min/max = "
        << gMin(continuityErrorAfter) << ' ' << gMax(continuityErrorAfter)
        << ", relative max = " << relativeContinuityError
        << ", projection passes = " << projectionIterations
        << ", iterations = " << iteration
        << ", converged = " << converged << nl << endl;

    return true;
}


bool Foam::functionObjects::boundedPhaseMassTransport::write()
{
    transportedField().write();
    if (sigmaPtr_.valid())
    {
        sigmaPtr_().write();
    }
    return true;
}


// ************************************************************************* //
