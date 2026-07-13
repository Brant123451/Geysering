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
#include "fvmDdt.H"
#include "fvmDiv.H"

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
    rhoName_(dict.getOrDefault<word>("rho", "rho")),
    fluxResultName_
    (
        dict.getOrDefault<word>("fluxResult", fieldName_ + "MassFlux")
    ),
    schemesField_("unknown-schemesField"),
    tolerance_(1),
    nCorr_(0),
    resetOnStartUp_(false),
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
    dict.readIfPresent("rho", rhoName_);
    dict.readIfPresent("fluxResult", fluxResultName_);
    schemesField_ = dict.getOrDefault("schemesField", fieldName_);
    dict.readIfPresent("tolerance", tolerance_);
    dict.readIfPresent("nCorr", nCorr_);
    dict.readIfPresent("resetOnStartUp", resetOnStartUp_);

    return true;
}


bool Foam::functionObjects::boundedPhaseMassTransport::execute()
{
    volScalarField& field = transportedField();
    const volScalarField& rho =
        lookupObject<volScalarField>(rhoName_);
    const surfaceScalarField& phi =
        lookupObject<surfaceScalarField>(phiName_);

    if (rho.dimensions() != dimDensity)
    {
        FatalErrorInFunction
            << "Expected phase mass density dimensions " << dimDensity
            << " for " << rho.name() << ", found " << rho.dimensions()
            << exit(FatalError);
    }

    if (phi.dimensions() != dimMass/dimTime)
    {
        FatalErrorInFunction
            << "Expected phase mass flux dimensions " << dimMass/dimTime
            << " for " << phi.name() << ", found " << phi.dimensions()
            << exit(FatalError);
    }

    const word divScheme("div(phi," + schemesField_ + ")");
    scalar relaxCoeff = 0;
    mesh_.relaxEquation(schemesField_, relaxCoeff);

    bool converged = false;
    label iteration = 0;
    tmp<surfaceScalarField> tracerFlux;

    for (label corr = 0; corr <= nCorr_; ++corr)
    {
        fvScalarMatrix fieldEqn
        (
            fvm::ddt(rho, field)
          + fvm::div(phi, field, divScheme)
        );

        fieldEqn.relax(relaxCoeff);

        ++iteration;
        converged =
        (
            fieldEqn.solve(schemesField_).initialResidual() < tolerance_
        );
        tracerFlux = fieldEqn.flux();

        if (converged)
        {
            break;
        }
    }

    // The stock scalarTransport function object only applies this
    // variable-density MULES correction in its volume-phase branch.  Applying
    // it here bounds the compressible phase mass fraction conservatively.
    MULES::explicitSolve
    (
        rho,
        field,
        phi,
        tracerFlux.ref(),
        oneField(),
        zeroField()
    );

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
        << ", min/max = " << gMin(field) << ' ' << gMax(field)
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
