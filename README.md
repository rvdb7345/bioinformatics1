# bioinformatics1

In this repository, we implement the model described by Martínez et al., 2012 and use an evolutionary algorithm to fit the parameters that are present in the model. The data used to fit to was obtained by Kirsten Wesenhagen.

The data of Kirsten Wesenhagen is given in wesenhagen_data.csv. The data used by Martinez et al. is given in martinez_data.csv.

The repository consists of several files, below we find an overview of what does what:
The main files used for creating the results in the papers:
- parameter_fitting_selfadaptivesigma_fullmodel.py: uses an EA with a self-adaptive mutation step size to fit the mu's and sigma's of the system as well as CD40.
- parameter_fitting_selfadaptivesigma_fullmodel_inclkandl.py: uses an EA with a self-adaptive mutation step size to fit the mu's, sigma's, k's and lambda's of the system as well as CD40
- parameter_fitting_selfadaptivesigma_IRF4.py: uses an EA with a self-adaptive mutation step size to fit the mu, sigma and CD40 for the isolated formula for IRF4.
- parameter_fitting_selfadaptivesigma_IRF4_inclkandl.py: uses an EA with a self-adaptive mutation step size to fit the mu, sigma, k, lambda and CD40 for the isolated formula for IRF4.

Some files for visualisation:
- Bioinf1_plotting.py: creates the plots for the found parameter combinations for the different files described above.
- plot3DbetapIRF4.py: creates a 3D image of the solution landscape for fitting IRF4 and creates a rotating graph.

Supplementary files that were not used for the creation of the final results, but that were created along the way of development and could help for future research:
- model_forward_euler.py & model_forward_euler_ourparams.py: simply the implementation of the model described by Martinez et al. where the differential equations are solved using the forward Euler technique. Furthermore, it produces some phase plots and does sensitivity analysis. In also implements the BCRsubnetwork and the CD40subnetwork separately.
- param_fit_BCL6.py: fits BCL6 decoupled from the rest of the system. Uses the deap libary to implement a basic evolutionary algorithm.
- param_fit_BLIMP.py: fits BLIMP decoupled from the rest of the system. Uses the deap libary to implement a basic evolutionary algorithm.
- parameter_fitting.py: attemps to fit the solutions of the ODEs to the affymetrix data. It solved the ODEs for a given parameter set and evolves a population using an evolutionary algorithm.
- parameter_fitting_docx.py: uses a basic EA for fitting the isolated IRF4 formula having rewritten it to condensed variables.
- parameter_fitting_docx_adaptK.py: uses a basic EA for fitting the isolated IRF4 formula having rewritten it to condensed variables.
- parameter_fitting_docx_adaptK_allparameters.py: uses a basic EA for fitting the isolated IRF4 formula. Here we fit mu, sigma, k, lambda and CD40, so all involved parameters
- parameter_fitting_selfadaptivesigma_fullmodel_cd0br0.py: uses an EA with a self-adaptive mutation step size to fit the mu's, sigma's, k's and lambda's of the system. Additionally, instead of taking a constant value for CD40 and neglecting BCR, we now use the formulas for the respective signals and fit the bcr0 and cd0 parameters.

References:

Martínez, M. R., Corradin, A., Klein, U., Álvarez, M. J., Toffolo, G. M., di Camillo, B., Califano, A. & Stolovitzky, G. a. (2012). Proceedings of the National Academy of Sciences   of the United States of America 109 , 2672-2677.
