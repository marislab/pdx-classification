# ROC Curve Plotting
#
# Gregory Way 2019
#
# Code for visualizing classification performance by ROC curve

library(ggplot2)

roc_file <- file.path("results", "full_roc_threshold_results.tsv")

roc_df <- readr::read_tsv(roc_file,
                          col_types = readr::cols(
                            .default = readr::col_double(),
                            gene = readr::col_character(),
                            shuffled = readr::col_character()
                            )
                          )

curve_colors <- c(
  "#1b9e77",
  "#d95f02",
  "#7570b3",
  "#737373",
  "#bdbdbd",
  "#d9d9d9"
)

curve_labels <- c(
  "TP53 False" = "TP53",
  "Ras False" = "Ras",
  "NF1 False" = "NF1",
  "TP53 True" = "TP53 Shuffled",
  "Ras True" = "Ras Shuffled",
  "NF1 True" = "NF1 Shuffled"
)
roc_df$model_groups <- paste(roc_df$gene, roc_df$shuffled)
roc_df$model_groups <- factor(roc_df$model_groups, levels = names(curve_labels))

ggplot(roc_df,
       aes(x = fpr,
           y = tpr,
           color = model_groups)) +
  coord_fixed() +
  geom_step(aes(linetype = shuffled), size = 0.4) +
  geom_segment(aes(x = 0,
                   y = 0,
                   xend = 1,
                   yend = 1),
               linetype = "dashed",
               color = "black") +
  scale_y_continuous(labels = scales::percent) +
  scale_x_continuous(labels = scales::percent) +
  scale_color_manual(name = "Models",
                     values = curve_colors,
                     labels = curve_labels) +
  scale_linetype_manual(name = "Data",
                        values = c("solid",
                                   "dashed"),
                        labels = c("True" = "Shuffled",
                                   "False" = "Real")) +
  xlab("False Positive Rate") +
  ylab("True Positive Rate") +
  theme_bw()
