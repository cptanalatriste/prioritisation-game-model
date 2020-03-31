library(tidyverse)

data_file <- read_csv("performance.csv")
all_performance_exp <- data_file %>% 
  mutate(EquilibriumId = str_c("(", EquilibriumId, ")"))  %>% 
  mutate(Parameters = replace_na(Parameters, ""))  %>% 
  mutate(ProcessCode = recode(Process, `Distributed Prioritisation` = "DP", Gatekeeper = "GK",
                              `Assessor-Throttling` = "AT")) %>%
  mutate(ConfigurationDesc = str_c(ProcessCode, Parameters, EquilibriumId, sep = " ")) %>%
  mutate(Prioritisation_Process = factor(ConfigurationDesc, levels = unique(ConfigurationDesc)))   

exp_full_bandwidth <- filter(all_performance_exp, Scenario == "Full Bandwidth")
exp_reduced_bandwith <- filter(all_performance_exp, Scenario == "Reduced Bandwidth")

data_for_plot <- exp_full_bandwidth

ggplot(data = data_for_plot) + 
  geom_bar(mapping = aes(x = Prioritisation_Process, y = Equilibrium_Fixes, fill = Process),
           stat = "identity") +
  theme_classic(base_size = font_size)