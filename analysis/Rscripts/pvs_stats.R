#_______________________________________________________________________________
#
# In this script we will test our hypothesis that PVS are
# associated with higher relevance in our brain-age estimation.
#
#_______________________________________________________________________________

library(tidyverse)
library(dplyr)
library(patchwork)
library(ggplot2)
library(cowplot)
library(rstatix)
library(see)
# Functions:

## Gives count, mean, standard deviation, standard error of the mean, and confidence interval (default 95%).
##   data: a data frame.
##   measurevar: the name of a column that contains the variable to be summariezed
##   groupvars: a vector containing names of columns that contain grouping variables
##   na.rm: a boolean that indicates whether to ignore NA's
##   conf.interval: the percent range of the confidence interval (default is 95%)

summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                      conf.interval=.95, .drop=TRUE) {
  library(plyr)
  
  # New version of length which can handle NA's: if na.rm==T, don't count them
  length2 <- function (x, na.rm=FALSE) {
    if (na.rm) sum(!is.na(x))
    else       length(x)
  }
  
  # This does the summary. For each group's data frame, return a vector with
  # N, mean, and sd
  datac <- ddply(data, groupvars, .drop=.drop,
                 .fun = function(xx, col) {
                   c(N    = length2(xx[[col]], na.rm=na.rm),
                     mean = mean   (xx[[col]], na.rm=na.rm),
                     sd   = sd     (xx[[col]], na.rm=na.rm)
                   )
                 },
                 measurevar
  )
  
  # Rename the "mean" column
  datac <- rename(datac, c("mean" = measurevar))
  
  datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean
  
  # Confidence interval multiplier for standard error
  # Calculate t-statistic for confidence interval:
  # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
  ciMult <- qt(conf.interval/2 + .5, datac$N-1)
  datac$ci <- datac$se * ciMult
  
  return(datac)
}

#_______________________________________________________________________________

# Colors:
green = "#438041"
redlight = "#dd7373"
grey = "#4C4C4B"
yellow = "#D1B74C"
blue = "#499293"
lightgreen = "#AFD099"
darkred = "#AA1926"
darkblue = "#44546A"
black = "#2C3531"


# ______________________________________________________________________________

# get path to data
localpath <- getwd()
projpath <- dirname(dirname(localpath)) 
respath <- paste0(projpath,"/results/PVS/")

# load in data and select only subjects with existing pvs
df_original_t1_comb <- read.csv(paste0(respath,"T1-FLAIR.PVS_t1-subens.csv"))
df_t1_comb <- filter(df_original_t1_comb, n_pvs_voxel > 0)
df_original_flair_comb <- read.csv(paste0(respath,"T1-FLAIR.PVS_flair-subens.csv"))
df_flair_comb <- filter(df_original_flair_comb, n_pvs_voxel > 0)

# load in WML data for figure
df_original <- read.csv(paste0(projpath,"/results/WML/WMlesion.csv"))
df <- filter(df_original, n_wml_voxel > 0)
df_n30 <- filter(df, n_wml_voxel > 30)

# Average relevance related to PVS
df_t1_comb1 <- data.frame(df_t1_comb$a_pvs_voxel,
                  rep("PVS",length(df_t1_comb$a_pvs_voxel)))
names(df_t1_comb1)[1] <- "a"
names(df_t1_comb1)[2] <- "var"


df_t1_comb2 <- data.frame(df_t1_comb$a_voxel,
                  rep("Expected",length(df_t1_comb$ap_voxel)))
names(df_t1_comb2)[1] <- "a"
names(df_t1_comb2)[2] <- "var"

df_flair_comb1 <- data.frame(df_flair_comb$a_pvs_voxel,
                          rep("PVS",length(df_flair_comb$a_pvs_voxel)))
names(df_flair_comb1)[1] <- "a"
names(df_flair_comb1)[2] <- "var"

df_flair_comb2 <- data.frame(df_flair_comb$ap_voxel,
                          rep("Expected",length(df_flair_comb$ap_voxel)))
names(df_flair_comb2)[1] <- "a"
names(df_flair_comb2)[2] <- "var"


df_t1_comb1<-rbind(df_t1_comb1,df_t1_comb2)
df_flair_comb1<-rbind(df_flair_comb1,df_flair_comb2)

summary_t1_comb <- summarySE(data=df_t1_comb1,
                     measurevar="a",
                     groupvars="var",
                     na.rm=FALSE, conf.interval=.95)

summary_flair_comb <- summarySE(data=df_flair_comb1,
                             measurevar="a",
                             groupvars="var",
                             na.rm=FALSE, conf.interval=.95)

# for WML 
df1 <- data.frame(df_n30$a_wml_voxel,
                  rep("WMH",length(df_n30$a_wml_voxel)))
names(df1)[1] <- "a"
names(df1)[2] <- "var"


df2 <- data.frame(df_n30$a_voxel,
                  rep("Expected",length(df_n30$a_wml_voxel)))
names(df2)[1] <- "a"
names(df2)[2] <- "var"

df1<-rbind(df1,df2)


# Average relevance related to PVS
df_t1_comb1 <- data.frame(df_t1_comb$a_pvs_voxel,
                          rep("PVS",length(df_t1_comb$a_pvs_voxel)))
names(df_t1_comb1)[1] <- "a"
names(df_t1_comb1)[2] <- "var"


df_t1_comb2 <- data.frame(df_t1_comb$a_voxel,
                          rep("Expected",length(df_t1_comb$a_voxel)))
names(df_t1_comb2)[1] <- "a"
names(df_t1_comb2)[2] <- "var"

df_flair_comb1 <- data.frame(df_flair_comb$a_pvs_voxel,
                             rep("PVS",length(df_flair_comb$a_pvs_voxel)))
names(df_flair_comb1)[1] <- "a"
names(df_flair_comb1)[2] <- "var"

df_flair_comb2 <- data.frame(df_flair_comb$a_voxel,
                             rep("Expected",length(df_flair_comb$a_voxel)))
names(df_flair_comb2)[1] <- "a"
names(df_flair_comb2)[2] <- "var"


df_t1_comb1<-rbind(df_t1_comb1,df_t1_comb2)
df_t1_comb1 <- df_t1_comb1 %>%
  mutate(a_transformed = log1p(a + 1))

df_flair_comb1<-rbind(df_flair_comb1,df_flair_comb2)
df_flair_comb1 <- df_flair_comb1 %>%
  mutate(a_transformed = log1p(a + 1))

# for WML 
df1 <- df1 %>%
  mutate(a_transformed = log1p(a + 1))

# exclude outlier in df1
df1 <- subset(df1, a_transformed < max(a_transformed))

summary_WML <- summarySE(data=df1,
                         measurevar="a_transformed",
                         groupvars="var",
                         na.rm=FALSE, conf.interval=.95)

summary_t1_comb <- summarySE(data=df_t1_comb1,
                             measurevar="a_transformed",
                             groupvars="var",
                             na.rm=FALSE, conf.interval=.95)

summary_flair_comb <- summarySE(data=df_flair_comb1,
                                measurevar="a_transformed",
                                groupvars="var",
                                na.rm=FALSE, conf.interval=.95)

y_min_all <- min(c(min(df_flair_comb1$a_transformed),min(df_t1_comb1$a_transformed)))
y_max_all <- max(c(max(df_flair_comb1$a_transformed),max(df_t1_comb1$a_transformed)))

ymin <- y_min_all - y_min_all *0.00000001
ymax <- y_max_all + y_max_all * 0.00000001

nudge_value <- 0.25  # Adjust this value to move the violins to the right

p1_t1 <- ggplot(df_t1_comb1, aes(x = factor(var), y = a_transformed)) +
  xlab("") + 
  theme_classic() +
  geom_hline(yintercept = log1p(1+0), colour =grey, linetype="dashed") +
  geom_violinhalf(aes(colour = var, fill = var), alpha = 0.5, scale = "width", position = position_nudge(x = nudge_value)) + 
  geom_jitter(aes(colour = var), size = 1.5, alpha = 0.8, shape = 16, position = position_jitter(width = 0.21)) +
  geom_errorbar(data = summary_t1_comb, width = .15, aes(ymin = a_transformed - ci, ymax = a_transformed + ci), colour = "black") +
  geom_point(data = summary_t1_comb, size = 2, colour = "black") +
  theme(legend.position = "none") +
  scale_color_manual(values = c(darkblue, darkred)) +
  scale_fill_manual(values = c(darkblue, darkred)) + 
  labs(y = "Relevance per voxel in log(x+2)", subtitle = "T1") +
  ylim(ymin,ymax)

p1_flair <- ggplot(df_flair_comb1, aes(x = factor(var), y = a_transformed)) +
  xlab("") + 
  theme_classic() +
  geom_hline(yintercept = log1p(1+0), colour =grey, linetype="dashed") +
  geom_violinhalf(aes(colour = var, fill = var), alpha = 0.5, scale = "width", position = position_nudge(x = nudge_value)) + 
  geom_jitter(aes(colour = var), size = 1.5, alpha = 0.8, shape = 16, position = position_jitter(width = 0.21)) +
  geom_errorbar(data = summary_flair_comb, width = .15, aes(ymin = a_transformed - ci, ymax = a_transformed + ci), colour = "black") +
  geom_point(data = summary_flair_comb, size = 2, colour = "black") +
  theme(legend.position = "none") +
  scale_color_manual(values = c(darkblue, darkred)) +
  scale_fill_manual(values = c(darkblue, darkred)) + 
  labs(y = "", subtitle = "FLAIR") +
  ylim(ymin,ymax)

p1_WML <- ggplot(df1, aes(x = factor(var), y = a_transformed)) +
  xlab("") + 
  theme_classic() +
  geom_hline(yintercept = log1p(1+0), colour =grey, linetype="dashed") +
  geom_jitter(aes(colour = var), size = 1.5, alpha = 0.8, shape = 16, position = position_jitter(width = 0.21)) +
  geom_violinhalf(data = subset(df1, var == "WMH"), colour = yellow, fill = yellow, alpha = 0.5, scale = "width", position = position_nudge(x = nudge_value)) + 
  geom_errorbar(data = summary_WML, width = .15, aes(ymin = a_transformed - ci, ymax = a_transformed + ci), colour = "black") +
  geom_point(data = summary_WML, size = 2, colour = "black") +
  theme(legend.position = "none") +
  scale_color_manual(values = c(darkblue, yellow)) +
  scale_fill_manual(values = c(darkblue, yellow)) + 
  labs(y = "", subtitle = "FLAIR") 


# Combine the plots
fig_pvs_wml_avg = ggdraw() +
  draw_plot(p1_t1, x=0, y=0,width = 1/3, height = 1)+
  draw_plot(p1_flair, x=1/3, y=0,width = 1/3, height = 1)+
  draw_plot(p1_WML, x=2/3, y=0,width = 1/3, height = 1)+
  draw_plot_label(c("a","b","c"), x=c(0,1/3,2/3),y=c(1,1,1),size=18)

# as svg
ggsave(filename = paste0(respath, "figure5.svg"), plot = fig_pvs_wml_avg,
       units="in", width=9, height=4, dpi=400)
# as png
ggsave(filename = paste0(respath, "figure5.png"), plot = fig_pvs_wml_avg,
       units="in", width=9, height=4, dpi=400)