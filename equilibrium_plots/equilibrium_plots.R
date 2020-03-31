library(tidyverse)

data_file <- read_csv("equilibria.csv")
all_equilibria <- data_file %>% 
  mutate(Parameters = replace_na(Parameters, ""))  %>% 
  mutate(Configuration = str_c(Process, Parameters, sep = " "))

reduced_bandwith <- filter(all_equilibria, Scenario == "Reduced Bandwidth")

ggplot(data = reduced_bandwith) + 
  geom_bar(mapping = aes(x = Configuration, y = Probability, fill = Strategy),
           position = "dodge", stat = "identity")