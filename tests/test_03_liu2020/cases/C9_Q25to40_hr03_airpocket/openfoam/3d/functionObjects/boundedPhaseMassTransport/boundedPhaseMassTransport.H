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

//- Transport a bounded mass fraction using phase fraction, phase density, and
//  phase mass flux.  The carrier density and its old-time value are built from
//  the solver-owned phase fields.  A phase-continuity source keeps constants
//  bounded, and its integral is exposed for a complete tracer budget.
class boundedPhaseMassTransport
:
    public fvMeshFunctionObject
{
    // Private Data

        word fieldName_;
        word phiName_;
        word alphaName_;
        word phaseRhoName_;
        word rhoResultName_;
        word fluxResultName_;
        word sourceResultName_;
        word schemesField_;
        scalar residualAlpha_;
        scalar tolerance_;
        label nCorr_;
        bool resetOnStartUp_;
        autoPtr<volScalarField> carrierRhoPtr_;
        autoPtr<volScalarField> tracerSourcePtr_;
        autoPtr<surfaceScalarField> tracerFluxPtr_;


    // Private Member Functions

        volScalarField& transportedField();

        volScalarField& updateCarrierDensity
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
