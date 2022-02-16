#----------------------------------------------------------------
# Sentiment analysis and clustering on two restaurants in London
#----------------------------------------------------------------

# Objective: giving recommendations to a restaurant with low ratings 

# Methodology: 
#     1. finding the restaurant's negative aspects 
#     2. using the reviews of a restaurant with outsanding reviews in order to understand what is expected by customers
# NB: both restaurants are in central London and serve similar food

# Reviews from both restaurants have been collected from opentable.com using selenium chrome.driver with python

# read the data

setwd("~/Desktop/Big data analytics")


wilton <- read.csv2("Wiltoncsv.csv")
abacus <- read.csv2("Abacuscsv.csv")

# ##################################### #
# RESTAURANT 1: WILTON -> GREAT REVIEWS #
# ##################################### #

# SENTIMENT ANALYSIS

# to lower case

wilton$Review <- tolower(wilton$Review)

# remove special characters
library(tm)
library("stringr")
wilton$Review <- str_replace_all(wilton$Review, "[^[:alnum:]]", " ")

# remove numbers

wilton$Review <- removeNumbers(wilton$Review)

# remove puntuation

wilton$Review <- removePunctuation(wilton$Review)

# remove stopwords and unwanted words

wilton$Review <- removeWords(wilton$Review, stopwords("english"))
wilton$Review <- removeWords(wilton$Review, c("s", "wilton", "wiltons","restaurant", "london", "food", "british", "service"))

# remove additional empty spaces if needed

wilton$Review <- stripWhitespace(wilton$Review)

head(wilton$Review) # to have a quick look at the structure of the reviews


# sentiment score analysis

library("tidytext")
library("dplyr")
library("sentimentr")
library("ggplot2")

wilton$sentscore <- sentiment(wilton$Review)
summary(wilton$sentscore)

# delete "empty" reviews

wilton <- filter(wilton, sentscore$word_count != "NA")


# separation of reviews into 4 groups: negative, neutral, positive, very positive (-> subgroup of positive, positive also includes very positive reviews)
str(wilton)
negative <- filter(wilton, sentscore$sentiment < -0.1)
neutral <- filter(wilton, sentscore$sentiment > -0.1 & sentscore$sentiment < 0.2)
positive <- filter(wilton, sentscore$sentiment > 0.2)
verypositive <- filter(wilton, sentscore$sentiment > 0.7)

str(verypositive)


# most frequent words and word cloud for each group of reviews

# Insight from very positive reviews

reviews_verypositive <- data.frame(verypositive$Review)

# Seperate riviews into words
data_by_word_verypositive <- reviews_verypositive %>% 
  mutate(linenumber=row_number()) %>%
  unnest_tokens(word, verypositive.Review)

# Count most common words 
head(data_by_word_verypositive %>%
  count(word, sort = TRUE), 11)


# Graph the most common words
p <- data_by_word_verypositive %>%
  count(word, sort = TRUE) %>%
  filter(n>77) %>% # , word != "wiltons"
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word,n)) + 
  geom_col() + 
  xlab("Words") + 
  ylab("Count") + 
  coord_flip()
p + ggtitle("Most Popular Words")


# Wordcloud

library(wordcloud)
vp <- as.vector(t(as.matrix(verypositive$Review)))
set.seed(1)
wordcloud(vp, max.words = 100, random.order=FALSE, colors = brewer.pal(8, "Dark2"),  rot.per=0)


# Insight from very negative reviews

reviews_negative <- data.frame(negative$Review)

# Seperate tweets into words
data_by_word_negative <- reviews_negative %>% 
  mutate(linenumber=row_number()) %>%
  unnest_tokens(word, negative.Review)

# Count most common words 
head(data_by_word_negative %>%
       count(word, sort = TRUE), 11)


# Graph the most common words
p <- data_by_word_negative %>%
  count(word, sort = TRUE) %>%
  filter(n>5) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word,n)) + 
  geom_col() + 
  xlab("Words") + 
  ylab("Count") + 
  coord_flip()
p + ggtitle("Most Popular Words")

# Wordcloud

n <- as.vector(t(as.matrix(negative$Review)))
set.seed(1)
wordcloud(n, max.words = 100, random.order=FALSE, colors = brewer.pal(8, "Dark2"),  rot.per=0)


# CLUSTERING

# 1. POSITIVE REVIEWS 

vp <- removeWords(vp, c("excellent","great","good","always","wonderful","special","place","lovely","well","outstanding","evening","perfect","delicious","will","best","staff","really","one"))

clusterdata <- Corpus(VectorSource(vp))
matrix <- TermDocumentMatrix(clusterdata)
sparse <-removeSparseTerms(matrix, sparse=0.95)
dataframe <- as.data.frame(as.matrix(sparse))
dataframe.scale <- scale(dataframe)
dataframe.dist <- dist(dataframe.scale, method = "euclidean")
dataframe.fit <-hclust(dataframe.dist, method="ward.D2")
plot(dataframe.fit, main="Cluster- very positive reviews Wilton")

groups <- cutree(dataframe.fit, k=10)
plot(dataframe.fit, main="Cluster - very positive reviews Wilton")
rect.hclust(dataframe.fit, k=10, border="red")

# 2. NEGATIVE REVIEWS

n <- removeWords(n, c("never"))
n <- removeWords(n, stopwords("en"))

clusterdata_neg <- Corpus(VectorSource(n))
matrix_neg <- TermDocumentMatrix(clusterdata_neg)
sparse_neg <-removeSparseTerms(matrix_neg, sparse=0.95)
dataframe_neg <- as.data.frame(as.matrix(sparse_neg))
dataframe.scale_neg <- scale(dataframe_neg)
dataframe.dist_neg <- dist(dataframe.scale_neg, method = "euclidean")
dataframe.fit_neg <-hclust(dataframe.dist_neg, method="ward.D2")
plot(dataframe.fit_neg, main="Cluster - negative reviews Wilton")

groups <- cutree(dataframe.fit_neg, k=3)
plot(dataframe.fit_neg, main="Cluster - negative reviews Wilton")
rect.hclust(dataframe.fit_neg, k=3, border="red")


# ##################################### #
# RESTAURANT 2: ABACUS -> LOWER REVIEWS #
# ##################################### #

# SENTIMENT ANALYSIS

# to lower case

abacus$Review <- tolower(abacus$Review)

# remove special characters

library("stringr")
abacus$Review <- str_replace_all(abacus$Review, "[^[:alnum:]]", " ")

# remove numbers

abacus$Review <- removeNumbers(abacus$Review)

# remove puntuation

abacus$Review <- removePunctuation(abacus$Review)

# remove stopwords and unwanted words

abacus$Review <- removeWords(abacus$Review, stopwords("english"))
abacus$Review <- removeWords(abacus$Review, c("s", "abacus","restaurant", "london", "food", "british","t"))

# remove additional empty spaces if needed

abacus$Review <- stripWhitespace(abacus$Review)

head(abacus$Review) # to have a quick look at the structure of the reviews


# sentiment score analysis

library("tidytext")
library("dplyr")
library("sentimentr")

abacus$sentscore <- sentiment(abacus$Review)
summary(abacus$sentscore)

# delete "empty" reviews

abacus <- filter(abacus, sentscore$word_count != "NA")


# separation of reviews into 4 groups: negative, neutral, positive, very positive (-> subgroup of positive, positive also includes very positive reviews)
str(abacus)
negative_abacus <- filter(abacus, sentscore$sentiment < 0.2)
neutral_abacus <- filter(abacus, sentscore$sentiment > 0.2 & sentscore$sentiment < 0.5)
positive_abacus <- filter(abacus, sentscore$sentiment > 0.5)
verypositive_abacus <- filter(abacus, sentscore$sentiment > 0.7)

str(verypositive_abacus)


# most frequent words and word cloud for each group of reviews

# Insight from very positive reviews

reviews_verypositive_abacus <- data.frame(verypositive_abacus$Review)

# Seperate riviews into words
data_by_word_verypositive_abacus <- reviews_verypositive_abacus %>% 
  mutate(linenumber=row_number()) %>%
  unnest_tokens(word, verypositive_abacus.Review)

# Count most common words 
head(data_by_word_verypositive_abacus %>%
       count(word, sort = TRUE), 11)


# Graph the most common words
p <- data_by_word_verypositive_abacus %>%
  count(word, sort = TRUE) %>%
  filter(n>8) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word,n)) + 
  geom_col() + 
  xlab("Words") + 
  ylab("Count") + 
  coord_flip()
p + ggtitle("Most Popular Words")

# Wordcloud

vp_abacus <- as.vector(t(as.matrix(verypositive_abacus$Review)))
set.seed(1)
wordcloud(vp_abacus, max.words = 100, random.order=FALSE, colors = brewer.pal(8, "Dark2"),  rot.per=0)


# Insight from negative reviews

reviews_negative_abacus <- data.frame(negative_abacus$Review)

# Seperate tweets into words
data_by_word_negative_abacus <- reviews_negative_abacus %>% 
  mutate(linenumber=row_number()) %>%
  unnest_tokens(word, negative_abacus.Review)

# Count most common words 
head(data_by_word_negative_abacus %>%
       count(word, sort = TRUE), 11)


# Graph the most common words
p <- data_by_word_negative_abacus %>%
  count(word, sort = TRUE) %>%
  filter(n>4) %>% # , word != "wiltons"
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word,n)) + 
  geom_col() + 
  xlab("Words") + 
  ylab("Count") + 
  coord_flip()
p + ggtitle("Most Popular Words")


# Wordcloud

n_abacus <- as.vector(t(as.matrix(negative_abacus$Review)))
set.seed(1)
wordcloud(n_abacus, max.words = 100, random.order=FALSE, colors = brewer.pal(8, "Dark2"),  rot.per=0)


# CLUSTERING

# 1. POSITIVE REVIEWS 

clusterdata_abacus <- Corpus(VectorSource(vp_abacus))
matrix_abacus <- TermDocumentMatrix(clusterdata_abacus)
sparse_abacus <-removeSparseTerms(matrix_abacus, sparse=0.95)
dataframe_abacus <- as.data.frame(as.matrix(sparse_abacus))
dataframe.scale_abacus <- scale(dataframe_abacus)
dataframe.dist_abacus <- dist(dataframe.scale_abacus, method = "euclidean")
dataframe.fit_abacus <-hclust(dataframe.dist_abacus, method="ward.D2")
plot(dataframe.fit_abacus, main="Cluster - very positive reviews Abacus")

groups2 <- cutree(dataframe.fit_abacus, k=10)
plot(dataframe.fit_abacus, main="Cluster - very positive reviews Abacus")
rect.hclust(dataframe.fit_abacus, k=10, border="red")

# 2. NEGATIVE REVIEWS

n_abacus <- removeWords(n_abacus, c("lunch","pub","grub","will","door","sure","going","seemed","open","else","day","one","just","get","also","enough","went","pre","think","makes","still","came","due","much","hour","city","people","gets","lunches","well","friday","must","music","got","later","order"))

clusterdata_neg_abacus <- Corpus(VectorSource(n_abacus))
matrix_neg_abacus <- TermDocumentMatrix(clusterdata_neg_abacus)
sparse_neg_abacus <-removeSparseTerms(matrix_neg_abacus, sparse=0.95)
dataframe_neg_abacus <- as.data.frame(as.matrix(sparse_neg_abacus))
dataframe.scale_neg_abacus <- scale(dataframe_neg_abacus)
dataframe.dist_neg_abacus <- dist(dataframe.scale_neg_abacus, method = "euclidean")
dataframe.fit_neg_abacus <- hclust(dataframe.dist_neg_abacus, method="ward.D2")
plot(dataframe.fit_neg_abacus, main="Cluster - negative reviews Abacus")

groups2 <- cutree(dataframe.fit_neg_abacus, k=4)
plot(dataframe.fit_neg_abacus, main="Cluster - negative reviews Abacus", cex=0.8)
rect.hclust(dataframe.fit_neg_abacus, k=4, border="red")



