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

//- Transport a phase mass fraction by advancing the conserved tagged-mass
//  density sigma = alpha*rho*s.  The supplied phase mass flux is projected
//  onto the discrete phase-continuity equation, then
//  fvm::ddt(sigma) + fvm::div(phi/(alpha*rho)_f, sigma) is solved.  The
//  fraction field s is recovered only for output, boundary conditions and
//  arrival diagnostics.  No clamp/clear and no carrier-continuity source.
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
        word sigmaResultName_;
        word fluxResultName_;
        word sourceResultName_;
        word schemesField_;
        scalar residualAlpha_;
        scalar tolerance_;
        scalar boundsTolerance_;
        scalar continuityTolerance_;
        label nCorr_;
        label nNonOrthCorr_;
        label nProjectionCorr_;
        bool resetOnStartUp_;
        autoPtr<volScalarField> potentialPtr_;
        autoPtr<volScalarField> inventoryRhoPtr_;
        autoPtr<volScalarField> sigmaPtr_;
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

        volScalarField& conservedDensity
        (
            const volScalarField& alpha,
            const volScalarField& phaseRho,
            const volScalarField& field
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
