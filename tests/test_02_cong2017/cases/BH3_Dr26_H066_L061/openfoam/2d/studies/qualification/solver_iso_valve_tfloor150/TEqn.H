{
    fvScalarMatrix TEqn
    (
        fvm::ddt(rho, T) + fvm::div(rhoPhi, T) - fvm::Sp(contErr, T)
      - fvm::laplacian(turbulence.alphaEff(), T)
      + (
             divUp()// - contErr/rho*p
          +  (fvc::ddt(rho, K) + fvc::div(rhoPhi, K))() - contErr*K
        )
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

    // Numerical safeguard for the deliberately severe outcome-forcing case.
    // The unbounded branch produced isolated negative cell temperatures during
    // the post-eruption high-speed jet, causing correctThermo() to abort before
    // it could write a recoverable checkpoint.  150 K is below the lowest
    // finite temperature previously observed in the accepted trajectory
    // (~169 K), so normal cells are untouched.  Every activation is logged and
    // this solver must remain labelled calibrated/numerically stabilized.
    const scalar temperatureFloor = 150.0;
    const scalar rawTMin = gMin(T);

    if (rawTMin < temperatureFloor)
    {
        label clippedCells = 0;
        const scalarField& TCells = T.primitiveField();

        forAll(TCells, celli)
        {
            if (TCells[celli] < temperatureFloor)
            {
                ++clippedCells;
            }
        }

        reduce(clippedCells, sumOp<label>());

        Info<< "temperatureFloor150: rawMin=" << rawTMin
            << " K, clippedCells=" << clippedCells << nl;

        T.max(temperatureFloor);
        T.correctBoundaryConditions();
    }

    mixture.correctThermo();
    mixture.correct();
}
