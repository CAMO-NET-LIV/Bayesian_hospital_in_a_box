library(hiabR)
library(future)
library(progressr)
library(tidyverse)

venv_hiab_path <- "hiab/"
install_hospital_in_a_box(env_name = venv_hiab_path)
check_hospital_in_a_box(env_name = venv_hiab_path)

n_sims <- 50
n_specimens <- 1e5
sim_seed <- 123

plan(multisession, workers = availableCores()-2)

n_specimens <- make_param_list(value = rep(n_specimens, n_sims),
                               name = "n_specimens",
                               type = "const")
mla_rec <- make_param_list(mean = rep(23, n_sims), 
                       name = "MLA_reception_inter_task_time",
                       type = "exp")
bms <- make_param_list(mean = rep(23, n_sims), 
                       name = "BMS_inter_task_time",
                       type = "exp")

sweep <- seq.int(from=1, to=150, length.out = n_sims)
sweep <- round(sweep)
mla_bc <- make_param_list(mean = sweep, 
                       name = "MLA_blood_cultures_inter_task_time",
                       type = "exp")

s <- with_progress(
  sim_wrapper(n_samples = n_sims,
            mc_iterations = 1,
            override_params = list(n_specimens, mla_rec, bms, mla_bc),
            sim_seed = as.integer(123),
            env_name = venv_hiab_path
            )
)

s_data <- s %>%
  map(\(x) x[[1]])

map2(s_data, seq(n_sims), \(x, y) {
  readr::write_csv(x, file.path("outputs", paste0("sim", y, ".csv")))
  })


