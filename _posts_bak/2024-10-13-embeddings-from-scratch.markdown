---
layout: post
title: "RAG from scratch: Representing words as vectors (Part 1)"
subtitle: "why dense vector search is important"
date: 2024-10-05 13:45:13 -0400
background: '/assets/img/posts/01_transformer_by_dalle.jpg'
---

# Intro

Large Language Models (LLMs) are great and have been very helpful in writing code, planning trips, learning new languages, and many other tasks. 

One problem with LLMs is that the knowledge embedded in their weights has a cutoff date and you're not guaranteed to have the latest and greatest information. Or, if the codebase you're working with, has been private during the LLM training then the model won't know what you're asking about.

A solution for this is Retrieval Augmented Generation (RAG), which I like to think of as fancy automated prompt engineering. It's a way of finding data relevant to the question you are asking and that information gets appended to the LLM prompt as additional context to help answer the query.

I believe knowledge retrieval will remain and increasingly important topic, which is why it's worth taking the time to understand how it's done today from first principles. It's still unclear if the massive models we're seeing used today will be the defacto way of running intelligent assistants in the future, or will we see a breakdown of specialization and maybe have some very tiny cognitive cores that do the reasoning (think <2B parameters in size). Whereas different functionalities will be delegated to expert models and knowledge won't be stored in the cognitive core weights, but retrieved from external databases.

## Dense Vector vs Lexical Search

Typical implementations of RAG today use dense vector search, i.e. documents get converted to embeddings and compared against the embeddings of the user prompt (query). This enables a more semantic comparison, looking at the "meaning" of the messages rather than an exact wording. In contrast old-school lexical search would be things like looking up keywords, exact metadata about a file.

If we want to find a customer file who's name is "Joe Shmoe", then it makes a lot more sense to simply do a word lookup -- we don't need to get fancy. However, in most LLM applications the inputs from users are unstructured natural language, which may have typos or mistakes. Due to this, the embeddings approach is very popular in modern RAG implementations and we'd use it for most cases where we want a more flexible understanding of user input. For example, even if a user asks for information using synonyms or related terms, dense embeddings can capture the semantic similarity and retrieve the relevant documents more accurately than traditional keyword-based searches.