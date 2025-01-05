# Laser pulse propagation in optical fiber

This notebook shows a simple, scalar implementation of the Fourier Pseudo Spectral Method <a href="#1">[1]</a> for solving the Nonlinear Schrödinger Equation <a href="#2">[2]</a>.

$\frac{\partial A}{\partial z}=-\frac{\alpha}{2}A+i \frac{\beta_2}{2} \frac{\partial^2 A}{\partial t^2}-i \gamma(|A|^2A)$

This nonlinear partial differential equation models how the envelope and phase of light pulse changes when propagating through a single mode optical fiber, when taking power attenuation ($\alpha$), group velocity dispersion ($\beta_2$)and waveguide nonlinearity ($\gamma$) causing self-phase modulation (SPM) into account. A is the slowly varying amplitude of the pulse envelope and t is measured in a frame of reference moving with the pulse at the group velocity $v_g$. The nonlinear Schrödinger equation (for optics) can be derived from the wave equation. However we can choose between two (Fourier transform) notations. The derivation with the negative one can be found in Ursula Keller's book <a href="#3">[3]</a>. I used this, because the scipy library uses the negative convention for the DFT. The plotting functions originally came from here <a href="#4">[4]</a>.

## References

<div id="1">[1] Wikimedia Foundation. (2024, November 28). Pseudo-spectral method. Wikipedia. https://en.wikipedia.org/wiki/Pseudo-spectral_method</div>
<div id="2">[2] Wikimedia Foundation. (2024, November 21). Nonlinear Schrödinger equation. Wikipedia. https://en.wikipedia.org/wiki/Nonlinear_Schr%C3%B6dinger_equation</div>
<div id="3">[3] Keller, U. (2023). Ultrafast lasers: A comprehensive introduction to fundamental principles with practical applications. Springer International Publishing.</div>
<div id="4">[4] Krarup, O. (n.d.). OLEKRARUP123/NLSE-vector-solver: Code for modelling the nonlinear Schrödinger equation for optical fiber pulse propagation. GitHub. https://github.com/OleKrarup123/NLSE-vector-solver</div>