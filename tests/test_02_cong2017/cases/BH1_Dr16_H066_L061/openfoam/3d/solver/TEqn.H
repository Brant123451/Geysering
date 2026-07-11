{
    // Internal-energy form from OpenFOAM Foundation commit 12dc41d0
    // (equivalent to totalInternalEnergy false).  Unlike the stock OpenCFD
    // v2512 equation, this deliberately excludes the explicit kinetic-energy
    // transport source while retaining p*div(U) pressure work.
    fvScalarMatrix TEqn
    (
        fvm::ddt(rho, T) + fvm::div(rhoPhi, T) - fvm::Sp(contErr, T)
      - fvm::laplacian(turbulence.alphaEff(), T)
      + p*fvc::div(fvc::absolute(phi, U))()()
       *(
           alpha1()/mixture.thermo1().Cv()()
         + alpha2()/mixture.thermo2().Cv()()
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
