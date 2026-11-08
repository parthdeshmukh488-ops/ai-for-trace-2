# References

Grouped by the part of the project they bear on. Each item is something the
implementation can be checked against or departs from deliberately.

## Test-time compute and search

* Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning.*
  The majority-vote baseline, which is the one most often beaten unfairly on a
  samples axis rather than a compute axis.
* Cobbe et al. (2021), *Training Verifiers to Solve Math Word Problems.* The
  outcome reward model and best-of-N.
* Lightman et al. (2023), *Let's Verify Step by Step.* Process supervision,
  and the step-label collection procedure whose hard-label bias F4 measures.
* Wang et al. (2024), *Math-Shepherd: Verify and Reinforce LLMs Step-by-step
  without Human Annotations.* Monte-Carlo estimation of step values, which is
  the estimator this project treats as ground truth rather than as a training
  signal.
* Snell et al. (2024), *Scaling LLM Test-Time Compute Optimally.* The
  compute-optimal allocation idea that F6's difficulty-aware arm tests, and
  the source of the framing this project is trying to put error bars on.
* Brown et al. (2024), *Large Language Monkeys: Scaling Inference Compute with
  Repeated Sampling.* The pass@k scaling behaviour that makes pass@k the right
  control in F7.
* Beeching et al. (2024), *Scaling Test-Time Compute with Open Models.*
  Diverse verifier tree search, and a practical account of beam search at the
  step level.
* Yao et al. (2023), *Tree of Thoughts.*
* Silver et al. (2017), *Mastering the Game of Go without Human Knowledge.*
  The PUCT formulation in SPEC §5, and the reason a sampled prior is a
  departure from it worth naming.

## Reward models, calibration and credit assignment

* Uesato et al. (2022), *Solving Math Word Problems with Process- and
  Outcome-based Feedback.*
* Murphy (1973), *A New Vector Partition of the Probability Score.* The
  reliability-resolution-uncertainty decomposition in SPEC §4.4.
* Guo et al. (2017), *On Calibration of Modern Neural Networks.* The source of
  the ECE conventions, including the binning issue that makes equal-mass bins
  the default here.
* Naeini et al. (2015), *Obtaining Well Calibrated Probabilities Using
  Bayesian Binning.*
* Geifman and El-Yaniv (2017), *Selective Classification for Deep Neural
  Networks.* The risk-coverage framing in F8.
* Kadavath et al. (2022), *Language Models (Mostly) Know What They Know.*

## Interval estimation

* Wilson (1927), *Probable Inference, the Law of Succession, and Statistical
  Inference.* The interval used throughout, chosen over the normal
  approximation for the reason SPEC §3.1 gives.
* Brown, Cai and DasGupta (2001), *Interval Estimation for a Binomial
  Proportion.* The paper that makes the case against the normal interval
  quantitatively; its coverage plots are what SPEC §3.4's simulation
  reproduces for this project's own estimator.
* Newcombe (1998), *Interval Estimation for the Difference Between
  Independent Proportions.* The difference interval that step advantage needs,
  and the reason subtracting endpoints is wrong.
* Armitage, McPherson and Rowe (1969), *Repeated Significance Tests on
  Accumulating Data.* Why the adaptive allocator's coverage must be measured
  rather than assumed.

## Reinforcement learning with verifiable rewards

* Schulman et al. (2017), *Proximal Policy Optimization Algorithms.* The
  clipped surrogate.
* Schulman (2020), *Approximating KL Divergence.* The `k3` estimator in
  SPEC §6.2 and why it is preferred to `-log(ratio)`.
* Shao et al. (2024), *DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models.* Group-relative policy optimisation as
  specified here.
* Liu et al. (2025), *Understanding R1-Zero-Like Training: A Critical
  Perspective.* The advantage-normalisation and length-normalisation biases
  that F7 measures rather than cites.
* Yue et al. (2025), *Does Reinforcement Learning Really Incentivize Reasoning
  Capacity in LLMs Beyond the Base Model?* The pass@k criterion that F7's
  central question is taken from.
* Hu et al. (2024), *Low-Rank Adaptation.* The `alpha/r` scaling convention
  that SPEC §6.1 insists on writing out.

## Evaluation hygiene

* Chen et al. (2021), *Evaluating Large Language Models Trained on Code.* The
  unbiased pass@k estimator in SPEC §7.1.
* Hendrycks et al. (2021), *Measuring Mathematical Problem Solving.*
* Brown et al. (2020), appendix C, *Language Models are Few-Shot Learners.*
  The 13-gram contamination methodology F9 follows.
* Zhang et al. (2024), *A Careful Examination of Large Language Model
  Performance on Grade School Arithmetic.* Why the completion probe is in F9
  and not only the overlap check.
* Dodge et al. (2019), *Show Your Work: Improved Reporting of Experimental
  Results.* Budget-aware reporting, which is what F6's compute axis is.
* Gelman and Loken (2013), *The Garden of Forking Paths.* The reason F1-F9
  record their hypotheses before the run.
