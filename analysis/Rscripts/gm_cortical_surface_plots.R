# ------------------------------------------------------------------------------
# Script to plot correlation results on cortical surface atlas
# ------------------------------------------------------------------------------

# Libraries
library(readr)
library(ggseg)
library(ggplot2)
library(cowplot)
library(dplyr)
library(ppcor)

# Paths
localpath <- getwd()
projpath <- dirname(dirname(localpath)) 
statspath <- file.path(projpath, "data/statsmaps/")
respath <- file.path(projpath, "results/GM/")

# Load merged dataset
df <- read_csv(file.path(statspath, "merged_cs_dktbase.csv"))


# Create label
df$label <- paste(df$hemisphere, df$structure_name, sep = "_")

# Prepare labels for plot functions
df$label <- gsub("left", "lh", df$label)
df$label <- gsub("right", "rh", df$label)

# Unique labels
unique_labels <- unique(df$label)

# Initialize correlation results for GMV
correlation_results_gmv <- data.frame(label = character(), correlation = numeric())

# Calculate correlation for GMV
for (label in unique_labels) {
  subset_data <- df[df$label == label, ]
  cor_test_result <- cor.test(subset_data$sum_relevance, subset_data$`gray_matter_volume_mm^3`)
  correlation_value <- cor_test_result$estimate
  p_value <- cor_test_result$p.value
  p_value_corrected <- p_value * length(unique_labels)  # Bonferroni correction
  
  new_row <- data.frame(label = label, r = correlation_value, p_value = p_value, p_value_corrected = p_value_corrected)
  correlation_results_gmv <- rbind(correlation_results_gmv, new_row)
}

# Initialize correlation results for CT
correlation_results_ct <- data.frame(label = character(), correlation = numeric())

# Calculate correlation for CT
for (label in unique_labels) {
  subset_data <- df[df$label == label, ]
  cor_test_result <- cor.test(subset_data$sum_relevance, subset_data$average_thickness_mm)
  correlation_value <- cor_test_result$estimate
  p_value <- cor_test_result$p.value
  p_value_corrected <- p_value * length(unique_labels)  # Bonferroni correction
  
  new_row <- data.frame(label = label, r = correlation_value, p_value = p_value, p_value_corrected = p_value_corrected)
  correlation_results_ct <- rbind(correlation_results_ct, new_row)
}

# Add significance columns
correlation_results_gmv$significant <- ifelse(correlation_results_gmv$p_value_corrected < 0.05, "yes", "no")
correlation_results_ct$significant <- ifelse(correlation_results_ct$p_value_corrected < 0.05, "yes", "no")

# Save results as CSV without row names
write.csv(correlation_results_gmv, file = file.path(respath, "correlation_results_cortical_gmv_dktbase.csv"), row.names = FALSE)
write.csv(correlation_results_ct, file = file.path(respath, "correlation_results_cortical_ct_dktbase.csv"), row.names = FALSE)

# Create plots
df_dk_thick <- correlation_results_ct
df_dk_gm <- correlation_results_gmv

# Filter significant results
df1_dk_gm <- subset(df_dk_gm, significant == "yes")
df1_dk_thick <- subset(df_dk_thick, significant == "yes")

# Plot for CT
p_dk_ct <- ggplot(df_dk_thick) +
  theme_void() +
  geom_brain(atlas = dk, position = position_brain(hemi ~ side), aes(fill = r)) +
  scale_fill_gradient2() +
  theme(text = element_text(size = 15)) +
  theme(plot.title = element_text(size = rel(1.5), face = 'bold'))

# Prepare second plot for CT
df1_dk_thick <- df1_dk_thick[!duplicated(df1_dk_thick), ]
df1_dk_thick <- df1_dk_thick[order(df1_dk_thick$r), ]
df1_dk_thick$label <- factor(df1_dk_thick$label, levels = df1_dk_thick$label)

p_dk_ct_2 <- ggplot(df1_dk_thick, aes(label, r)) +
  theme_classic() +
  geom_point(aes(colour = r), size = 3) +
  geom_segment(aes(xend = label, yend = r, colour = r, y = 0, x = label), linewidth = 1.5) +
  scale_colour_gradient2() +
  theme(text = element_text(size = 20)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 15)) +
  xlab("region") +
  theme(panel.grid.major = element_line(color = "grey", size = 0.3, linetype = 2)) +
  theme(legend.position = "none")

fig <- ggdraw() +
  draw_plot(p_dk_ct, x = 0, y = 0.5, width = 1, height = 0.5) +
  draw_plot(p_dk_ct_2, x = 0, y = 0, width = 1, height = 0.5)

# Save CT plots as PNG and SVG
ggsave(file.path(respath, "figure2.png"), plot = fig, device = "png", width = 25, height = 30, units = "cm", dpi = 500)
ggsave(file.path(respath, "figure2.svg"), plot = fig, device = "svg", width = 25, height = 30, units = "cm", dpi = 500)

# Plot for GMV
p_dk_gm <- ggplot(df_dk_gm) +
  theme_void() +
  geom_brain(atlas = dk, position = position_brain(hemi ~ side), aes(fill = r)) +
  scale_fill_gradient2() +
  theme(text = element_text(size = 15)) +
  theme(plot.title = element_text(size = rel(1.5), face = 'bold'))

# Prepare second plot for GMV
df1_dk_gm <- df1_dk_gm[!duplicated(df1_dk_gm), ]
df1_dk_gm <- df1_dk_gm[order(df1_dk_gm$r), ]
df1_dk_gm$label <- factor(df1_dk_gm$label, levels = df1_dk_gm$label)

p_dk_gm_2 <- ggplot(df1_dk_gm, aes(label, r)) +
  theme_classic() +
  geom_point(aes(colour = r), size = 3) +
  geom_segment(aes(xend = label, yend = r, colour = r, y = 0, x = label), size = 1.5) +
  scale_colour_gradient2() +
  theme(text = element_text(size = 20)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 15)) +
  xlab("region") +
  theme(panel.grid.major = element_line(color = "grey", size = 0.3, linetype = 2)) +
  theme(legend.position = "none")

fig <- ggdraw() +
  draw_plot(p_dk_gm, x = 0, y = 0.5, width = 1, height = 0.5) +
  draw_plot(p_dk_gm_2, x = 0, y = 0, width = 1, height = 0.5)

# Save GMV plots as PNG and SVG
ggsave(file.path(respath, "figure3.png"), plot = fig, device = "png", width = 25, height = 30, units = "cm", dpi = 500)
ggsave(file.path(respath, "figure3.svg"), plot = fig, device = "svg", width = 25, height = 30, units = "cm", dpi = 500)

# Overlap of labels between GMV and CT
labels_overlap <- intersect(df1_dk_gm$label, df1_dk_thick$label)

# Print the number of overlapping labels
cat("Number of labels in both GMV and CT:", length(labels_overlap), "\n")

