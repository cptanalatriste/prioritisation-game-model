library(tidyverse)

data_file <- read_csv("performance.csv")
all_performance_exp <- data_file %>% 
  mutate(EquilibriumId = str_c("(", EquilibriumId, ")"))  %>% 
  mutate(Parameters = replace_na(Parameters, ""))  %>% 
  mutate(ProcessCode = recode(Process, `Distributed Prioritisation` = "DP", Gatekeeper = "GK",
                              `Assessor-Throttling` = "AT")) %>%
  mutate(ConfigurationDesc = str_c(ProcessCode, Parameters, EquilibriumId, sep = " ")) %>%
  mutate(Prioritisation_Process = factor(ConfigurationDesc, 
                                         levels = unique(ConfigurationDesc)))   

all_performance_exp <- mutate(all_performance_exp, Prioritisation_Process = factor(Prioritisation_Process, levels = c("DP  (1)", "GK 50% (1)", "GK 10% (1)", "GK 0% (1)", "GK 0% (2)", "GK 0% (3)",	
                                                                                                             "GK 0% (4)",	"AT 3% (1)","AT 10% (1)",	"AT 20% (1)")))

exp_full_bandwidth <- filter(all_performance_exp, Scenario == "Full Bandwidth")
exp_reduced_bandwith <- filter(all_performance_exp, Scenario == "Reduced Bandwidth")

data_for_plot <- exp_full_bandwidth
font_size <- 20

ggplot(data = data_for_plot) + 
  geom_bar(mapping = aes(x = Prioritisation_Process, y = Equilibrium_Fixes, fill = Process),
           stat = "identity") +
  theme(text = element_text(size = font_size))
