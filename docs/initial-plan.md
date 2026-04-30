## GOAL
Your goal is to implement in Jax/Equinox the simplified atmospheric boundary layer (ABL) proposed by Lemarié et al. 2021 (https://doi.org/10.5194/gmd-14-543-2021, pdf available locally in `reference_paper/`) in JAX/Equinox. We also use as reference the FORTRAN implementation of ABL available locally from `nemo_abl1d_GMD_2020/NEMO_CODE/` into a JAX/Equinox stack.

## RULES
The code should implement strictly the equations available in Lemarié et al. 2021 pape, using robust and well tested numerical schemes. The code should be implemented in Jax, with proven differentiability for the entire solver (foreward and reverse mode autodiff). The code should include extensive tests and include all the demonstration set-uop described in sectioj 4 of the paper. The user should be able to reproduce easily the plots of the article with the Jax code.   

## STEPS

- read carefully the article Lemarié et al. 2021 (https://doi.org/10.5194/gmd-14-543-2021, pdf available locally in `reference_paper/`)
- scan the  FORTRAN implementation of ABL available locally from `nemo_abl1d_GMD_2020/NEMO_CODE/`
- prepare and write a plan in  `docs/mplementation_plan.md` 
- write a description of the contiunious equations to be implemenented in a `docs/continous_equations.md` 
- propose a first description of the discrtete algorithm you will implement  in `docs/discrete_implementation.md`
- proceed withe plan with a ralph loop. at the end of each iteration, the code should be pip installable, and fonctional, with tests corresponding to the current level of implementation. all code should be commited locally and pushed to the GH repos. large development should be done in dedicated branches and merged only when fonctional
- the README.mld should be in phase with the code before each commit (content, desription, CLI, algorithm, project status, ...)
- at the end of each iteration, keep a clear and short description of the project status in `docs/project_status.md`

