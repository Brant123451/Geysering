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

//- Transport a bounded mass fraction using a supplied phase mass density and
//  phase mass flux.  Unlike scalarTransport's compressible branch, the
//  implicit conservative solve is followed by variable-density MULES, so the
//  field remains in [0,1] without a non-conservative projection.
class boundedPhaseMassTransport
:
    public fvMeshFunctionObject
{
    // Private Data

        word fieldName_;
        word phiName_;
        word rhoName_;
        word fluxResultName_;
        word schemesField_;
        scalar tolerance_;
        label nCorr_;
        bool resetOnStartUp_;
        autoPtr<surfaceScalarField> tracerFluxPtr_;


    // Private Member Functions

        volScalarField& transportedField();


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
