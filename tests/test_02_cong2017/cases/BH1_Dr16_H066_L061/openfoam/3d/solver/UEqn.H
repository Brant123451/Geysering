    fvVectorMatrix UEqn
    (
        fvm::ddt(rho, U) + fvm::div(rhoPhi, U)
      - fvm::Sp(contErr, U)
      + MRF.DDt(rho, U)
      + turbulence.divDevRhoReff(U)
     ==
        fvOptions(rho, U)
    );

    if
    (
        valveActive
     && runTime.value() < valveOpeningDuration
    )
    {
        const scalar normalizedTime =
            min(max(runTime.value()/valveOpeningDuration, scalar(0)), scalar(1));
        const scalar openingSine =
            sin(0.5*constant::mathematical::pi*normalizedTime);
        const scalar areaFraction =
            max(valveMinimumAreaFraction, sqr(openingSine));
        const scalar lossCoefficient =
            max(scalar(0), 1.0/sqr(areaFraction) - 1.0);

        // Picard-linearized Forchheimer loss:
        //
        //   grad(p)_loss = 0.5*rho*K*|U|*U/L.
        //
        // Adding its non-negative coefficient to the momentum diagonal keeps
        // the source passive and couples the resistance into rAU/p_rgh.  The
        // exact source remains zero at zero velocity; no velocity floor,
        // pressure jump, or prescribed flow is used.
        scalarField& Udiag = UEqn.diag();
        const scalarField& cellVolumes = mesh.V();
        const labelList& valveCells =
            mesh.cellZones()[valveResistanceZoneID];

        for (const label celli : valveCells)
        {
            Udiag[celli] +=
                cellVolumes[celli]
               *0.5*rho[celli]*lossCoefficient*mag(U[celli])
               /valveResistanceLength;
        }

        if
        (
            runTime.timeIndex() <= runTime.startTimeIndex() + 2
         || runTime.writeTime()
        )
        {
            Info<< "Equivalent valve: A/A0=" << areaFraction
                << ", K=" << lossCoefficient << nl;
        }
    }

    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (pimple.momentumPredictor())
    {
        solve
        (
            UEqn
         ==
            fvc::reconstruct
            (
                (
                    mixture.surfaceTensionForce()
                  - ghf*fvc::snGrad(rho)
                  - fvc::snGrad(p_rgh)
                ) * mesh.magSf()
            )
        );

        fvOptions.correct(U);

        K = 0.5*magSqr(U);
    }
