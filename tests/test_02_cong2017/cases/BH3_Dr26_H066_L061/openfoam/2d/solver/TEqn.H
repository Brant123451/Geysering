{
    // Total-sensible-enthalpy form used by the audited H1 3D solver.
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

