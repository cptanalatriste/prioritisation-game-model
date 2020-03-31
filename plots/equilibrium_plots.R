library(tidyverse)

data_file <- read_csv("equilibria.csv")
all_equilibria <- data_file %>% 
  mutate(EquilibriumId = str_c("(", EquilibriumId, ")"))  %>% 
  mutate(Parameters = replace_na(Parameters, ""))  %>% 
  mutate(ProcessCode = recode(Process, `Distributed Prioritisation` = "DP", Gatekeeper = "GK",
                              `Assessor-Throttling` = "AT")) %>%
  mutate(ConfigurationDesc = str_c(ProcessCode, Parameters, EquilibriumId, sep = " ")) %>%
  mutate(Prioritisation_Process = factor(ConfigurationDesc, levels = unique(ConfigurationDesc)))   
  

full_bandwidth <- filter(all_equilibria, Scenario == "Full Bandwidth")
reduced_bandwith <- filter(all_equilibria, Scenario == "Reduced Bandwidth")

data_for_plot <- full_bandwidth
font_size <- 11

ggplot(data = data_for_plot) + 
  geom_bar(mapping = aes(x = Prioritisation_Process, y = Probability, fill = Strategy),
           position = bar_position, stat = "identity") +
  theme_classic(base_size = font_size)