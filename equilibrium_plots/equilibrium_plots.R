library(tidyverse)

data_file <- read_csv("equilibria.csv")
all_equilibria <- data_file %>% 
  mutate(Parameters = replace_na(Parameters, ""))  %>% 
  mutate(ProcessCode = recode(Process, `Distributed Prioritisation` = "DP", Gatekeeper = "GK",
                              `Assessor-Throttling` = "AT")) %>%
  mutate(Configuration = str_c(ProcessCode, Parameters, EquilibriumId, sep = " ")) 

reduced_bandwith <- filter(all_equilibria, Scenario == "Reduced Bandwidth")
full_bandwidth <- filter(all_equilibria, Scenario == "Full Bandwidth")

data_for_plot <- full_bandwidth
bar_position <- "fill"
ggplot(data = data_for_plot) + 
  geom_bar(mapping = aes(x = Configuration, y = Probability, fill = Strategy),
           position = bar_position, stat = "identity")