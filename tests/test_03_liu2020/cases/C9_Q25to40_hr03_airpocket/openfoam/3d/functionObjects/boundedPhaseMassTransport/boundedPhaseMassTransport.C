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
#include "addToRunTimeSelectionTable.H"
#include "fixedValueFvPatchField.H"
#include "fvcDdt.H"
#include "fvcDiv.H"
#include "fvmDdt.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "fvmSup.H"
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
    // Ensure old-time storage exists before the first ddt evaluation.
    field.oldTime();
    field.correctBoundaryConditions();

    const word divScheme("div(phi," + schemesField_ + ")");
    scalar relaxCoeff = 0;
    mesh_.relaxEquation(schemesField_, relaxCoeff);

    // Compressible phase-fraction transport must use the bounded ddt form
    // αρ*dds/dt, not ddt(αρs).  Otherwise independent α/ρ evolution from
    // MULES/thermo is misinterpreted as a change in the tag and erodes the
    // pocket inventory even when carrier continuity is projected.
    const volScalarField alphaRhoDdt(fvc::ddt(alpha, phaseRho));

    bool converged = false;
    label iteration = 0;
    tmp<surfaceScalarField> tracerFlux;

    for (label corr = 0; corr <= nCorr_; ++corr)
    {
        fvScalarMatrix fieldEqn
        (
            fvm::ddt(alpha, phaseRho, field)
          - fvm::Sp(alphaRhoDdt, field)
          + fvm::div(carrierFlux, field, divScheme)
         ==
          - fvm::ddt(residualAlpha_*phaseRho, field)
          + fvc::ddt(residualAlpha_*phaseRho, field)
        );

        fieldEqn.relax(relaxCoeff);

        ++iteration;
        const auto solverPerformance = fieldEqn.solve(schemesField_);
        converged = solverPerformance.finalResidual() < tolerance_;
        tracerFlux = fieldEqn.flux();
        tracerFlux.ref().oriented() = carrierFlux.oriented();

    }

    if (!converged)
    {
        FatalErrorInFunction
            << field.name() << " did not reach linear-solver tolerance "
            << tolerance_ << " after " << iteration << " solve(s)."
            << exit(FatalError);
    }

    const scalar fieldMin = gMin(field);
    const scalar fieldMax = gMax(field);
    if
    (
        fieldMin < -boundsTolerance_
     || fieldMax > 1 + boundsTolerance_
    )
    {
        FatalErrorInFunction
            << field.name() << " left [0,1]: min/max = "
            << fieldMin << ' ' << fieldMax
            << ", violations = " << max(-fieldMin, scalar(0))
            << ' ' << max(fieldMax - 1, scalar(0))
            << ", tolerance = " << boundsTolerance_ << nl
            << "Refusing to continue with a non-physical source tracer."
            << exit(FatalError);
    }

    updateInventoryDensity(alpha, phaseRho);

    // Discrete tagged-mass residual of ∫ alpha*rho*s after the bounded solve.
    // With the Sp(ddt(alpha,rho)) form this should stay near the projected
    // carrier-continuity residual rather than track artificial s erosion.
    const tmp<volScalarField> tracerSource
    (
        fvc::ddt(alpha, phaseRho, field) + fvc::div(tracerFlux())
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
                tracerFlux()
            )
        );
    }
    else
    {
        tracerFluxPtr_() = tracerFlux();
    }

    Log << type() << " execute: " << field.name()
        << ", min/max = " << fieldMin << ' ' << fieldMax
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
    return true;
}


// ************************************************************************* //
