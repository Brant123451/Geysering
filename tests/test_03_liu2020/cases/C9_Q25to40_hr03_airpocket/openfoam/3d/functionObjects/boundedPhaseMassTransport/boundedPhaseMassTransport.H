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

#ifndef Foam_functionObjects_boundedPhaseMassTransport_H
#define Foam_functionObjects_boundedPhaseMassTransport_H

#include "fvMeshFunctionObject.H"
#include "surfaceFields.H"
#include "volFields.H"

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                Class boundedPhaseMassTransport Declaration
\*---------------------------------------------------------------------------*/

//- Transport a phase mass fraction using the solver-owned phase fraction and
//  density time levels.  The supplied phase mass flux is projected onto the
//  discrete phase-continuity equation before an implicit upwind scalar solve.
//  A residual-phase deferred correction regularises cells in which the phase
//  vanishes without adding a physical tracer source.
class boundedPhaseMassTransport
:
    public fvMeshFunctionObject
{
    // Private Data

        word fieldName_;
        word phiName_;
        word alphaName_;
        word phaseRhoName_;
        word pName_;
        word potentialName_;
        word carrierFluxResultName_;
        word rhoResultName_;
        word fluxResultName_;
        word sourceResultName_;
        word schemesField_;
        scalar residualAlpha_;
        scalar tolerance_;
        scalar boundsTolerance_;
        scalar continuityTolerance_;
        label nCorr_;
        label nNonOrthCorr_;
        bool resetOnStartUp_;
        autoPtr<volScalarField> potentialPtr_;
        autoPtr<volScalarField> inventoryRhoPtr_;
        autoPtr<volScalarField> tracerSourcePtr_;
        autoPtr<surfaceScalarField> carrierFluxPtr_;
        autoPtr<surfaceScalarField> tracerFluxPtr_;


    // Private Member Functions

        volScalarField& transportedField();

        volScalarField& potential
        (
            const surfaceScalarField& phi,
            const volScalarField& p
        );

        volScalarField& updateInventoryDensity
        (
            const volScalarField& alpha,
            const volScalarField& phaseRho
        );


public:

    TypeName("boundedPhaseMassTransport");

    boundedPhaseMassTransport
    (
        const word& name,
        const Time& runTime,
        const dictionary& dict
    );

    virtual ~boundedPhaseMassTransport() = default;

    virtual bool read(const dictionary& dict);

    virtual bool execute();

    virtual bool write();
};

} // End namespace functionObjects
} // End namespace Foam

#endif

// ************************************************************************* //
