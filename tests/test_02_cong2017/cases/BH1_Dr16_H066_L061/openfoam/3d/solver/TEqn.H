{
    // Total-sensible-enthalpy form.  For each pure phase this is the standard
    // OpenFOAM transformation of total internal energy:
    //
    //   rho Cp DT/Dt + D(rho K)/Dt - dp/dt = 0.
    //
    // Using dp/dt instead of div(p U) avoids numerical cancellation of the
    // atmospheric pressure datum against p*div(U) in the low-Mach exterior.
    fvScalarMatrix TEqn
    (
        fvm::ddt(rho, T) + fvm::div(rhoPhi, T) - fvm::Sp(contErr, T)
      - fvm::laplacian(turbulence.alphaEff(), T)
      + (
            (fvc::ddt(rho, K) + fvc::div(rhoPhi, K))()
          - contErr*K
          - dpdt()
        )
       *(
           alpha1()/mixture.thermo1().Cp()()
         + alpha2()/mixture.thermo2().Cp()()
        )
     ==
        fvOptions(rho, T)
    );

    TEqn.relax();

    fvOptions.constrain(TEqn);

    TEqn.solve();

    fvOptions.correct(T);

    mixture.correctThermo();
    mixture.correct();
}
