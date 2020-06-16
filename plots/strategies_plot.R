library(tidyverse)

data_file <- read_csv("strategies.csv")

font_size <- 11

ggplot(data = data_file, aes(x = Inflation_Probability, y = Deflation_Probability, 
                             label = paste("Size = ", Cluster_Size))) + 
  geom_point(mapping = aes(color = Strategy, shape = Strategy), 
             size = 8) + 
  geom_text(vjust = 0, nudge_y = 0.03) +
  scale_shape_manual(values=seq(13,19)) +
  theme(legend.text=element_text(size=rel(1.0)))


