#CELL 1
import datetime
import json
import os
import requests
import re
from asknews_sdk import AskNewsSDK
import textwrap
import time
import anthropic
import numpy as np
## CONSTANTS
SUBMIT_PREDICTION = True # set to True to publish your predictions to Metaculus
FORECAST_TOURNAMENT = True # set to True to forecast all tournament questions
GET_NEWS = True # set to True to enable AskNews after entering ASKNEWS secrets
num_runs=5 # number of times to run the LLM
ONLY_NEW=0 # Only predict on new questions

# Environment variables
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
if GET_NEWS == True:
    ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
    ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")

## LLM Prompt

PROMPT_TEMPLATE = """
IMPORTANT: Assume that today is {today}.

You are a professional forecaster. You read the question very carefully and understand it.
You are hard working and don't take short cuts. You follow all instructions precisely. Your answer can be long. Make sure to take as much space as you need.
I will mark some instructions as IMPORTANT. Make sure that you pay special attention here and follow closely, as you had trouble following these in the past!
Thank you!

First, you state the question, background information, and resolution criteria.

Your question is:
{title}

Background:
{background}

Resolution criteria:
{resolution_criteria}

Make sure that you have properly understood the question, background, and resolution criteria.

IMPORTANT: Example of Resolution:
If there is a formula that is relevant for grading the question, state it explicitely. Provide an example for the individual factors making up the formula. Calculate how the question would be resolved.

You ALWAYS forecast questions according to the following procedure, you do not skip any step or substep:

1. State the current date, the date when the outcome to the question is known, and the number of days in between. Check whether there is reliable and current betting odds/metaculus probabilities on the question to form a prior. Take the betting odds and calculate an implied probability, while keeping in mind house edge.
If the question is an exact match (same question, same date, same resolution criteria, etc.) to the betting odds, do not change it further and just report it. In this case, you should be relatively certain about your estimate.
Otherwise, try to form a prior while considering the betting odds and continue with step 2.IMPORTANT: When considering predictions for different time frames, ALWAYS state the implied base rate on a monthly or yearly basis. Then form your prior based on that.
The closer the question can be matched to the betting odds question, the less should you change your opinion
with new information from steps 3+.

State your current estimate in % form and your current certainty [0-10].

2.If there are no betting odds/metaculus probabilities, form a prior from your general knowledge of the world. How likely is it that the question resolves positively in general?
How strong is your prior? If your prior is strong, don't let the next steps influence you too much. IMPORTANT: If your assistants disagree, don't rely on their info too strongly!

Before stating your prior, first state:
(a) The number of days until the outcome is known.
(b) What the outcome would be if nothing changed.
(c) Consider whether it is reasonable that the outcome would change in the given timeframe. What is your prior given year on year base rates?
(d) If the question relates to a specific date, consider the cases that 1/4th of the time was left and that 4 times the time was left. What would your prior be then? Note that oftentimes more time means more chances for something to happen. This may increase the probability.
(e) How many days would have to be left such that the question has a 50% probability of being resolved positively? Usually, more time makes it more likely that something happens.

State your current estimate in % form and your current certainty [0-10].

3. Use the news output from your research assistant and update the prior with the new information, depending on how relevant and trustworthy the information is. Consider whether sources may be biased. In general, there might be a slight pro left-wing bias in western sources and pro governemnt biases in general.

State your current estimate in % form and your current certainty [0-10].

4. Calculate X=min(12-current certainty,0.3 times your current estimate)
Consider that you have a bias towards things happening more often than they actually happen.
Account for it by lowering your estimate by X percentage points.

State your current estimate in % form and your current certainty [0-10].

Fine print:
{fine_print}

Your Metaculus assistant says:
{meta_assistant}

Average predictions from Metaculus Quarterly Cup:

{predictions_full}

Further information that may help to update the prior from your assistants. IMPORTANT: Be careful with this information if the assistants disagree!

Assistant 1:

{prior_info}

Assistant 2:

{prior_info2}

Your news sources say:
{summary_report}


IMPORTANT: The last thing you write is your final answer to the original question EXACTLY as: "Probability: ZZ%", ZZ in 0-100, integer (NO COMMA OR DOT), IMPORTANT: DO NOT FORGET THE % SIGN OR THE PREDICTION WON'T COUNT!

"""

## LLM Prompt

PROMPT_TEMPLATE_NUMERIC = """
IMPORTANT: Assume that today is {today}.

You are a professional forecaster. You read the question very carefully and understand it.
You are hard working and don't take short cuts. You follow all instructions precisely. Your answer can be long. Make sure to take as much space as you need.
I will mark some instructions as IMPORTANT. Make sure that you pay special attention here and follow closely, as you had trouble following these in the past!
Thank you!

First, you state the question, background information, and resolution criteria.

Your question is:
{title}

Background:
{background}

Resolution criteria:
{resolution_criteria}

Make sure that you have properly understood the question, background, and resolution criteria. You will answer a NUMERIC question, meaning that you give all of your answers as percentiles as so:
"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"

IMPORTANT: Sometimes the unit in which the question should be answered is unclear beacuse the question is poorly posed. Most of the mass of the question should lie between {scaling_min} and {scaling_max}.
Based on this, state the unit that you believe is most likely for this question. Answer the question in this unit going forth.

{lower_bound_message}
{upper_bound_message}

IMPORTANT: Example of Resolution:
If there is a formula that is relevant for grading the question, state it explicitely. Provide an example for the individual factors making up the formula. Calculate how the question would be resolved.

You ALWAYS forecast questions according to the following procedure, you do not skip any step or substep:

1. State the current date, the date when the outcome to the question is known, and the number of days in between. Check whether there is reliable and current betting odds/metaculus probabilities on the question to form a prior. Take the betting odds and calculate an implied probability, while keeping in mind house edge.
If the question is an exact match (same question, same date, same resolution criteria, etc.) to the betting odds, do not change it further and just report it. In this case, you should be relatively certain about your estimate.
Otherwise, try to form a prior while considering the betting odds and continue with step 2.IMPORTANT: When considering predictions for different time frames, ALWAYS state the implied base rate on a monthly or yearly basis. Then form your prior based on that.
The closer the question can be matched to the betting odds question, the less should you change your opinion
with new information from steps 2+.

State your current estimate and your current certainty [0-10].

2.If there are no betting odds/metaculus probabilities, form a prior from your general knowledge of the world.
How strong is your prior? If your prior is strong, don't let the next steps influence you too much. IMPORTANT: If your assistants disagree, don't rely on their info too strongly!

Before stating your prior, first state:
(a) The number of days until the outcome is known.
(b) What the outcome would be if nothing changed.
(c) Consider whether it is reasonable that the outcome would change in the given timeframe. What is your prior given year on year base rates?
(d) If the question relates to a specific date, consider the cases that 1/4th of the time was left and that 4 times the time was left. What would your prior be then? Note that oftentimes more
time means more chances for something to happen. This may increase the probability for change.

State your current estimate and your current certainty [0-10].

3. Use the news output from your research assistant and update the prior with the new information, depending on how relevant and trustworthy the information is.
Consider whether sources may be biased. In general, there might be a slight pro left-wing bias in western sources and pro governemnt biases in general.

State your current estimate and your current certainty [0-10].

Fine print:
{fine_print}

Your Metaculus assistant says:
{meta_assistant}

Average predictions from Metaculus Quarterly Cup:

{predictions_full}

Further information that may help to update the prior from your assistants. IMPORTANT: Be careful with this information if the assistants disagree!

Assistant 1:

{prior_info}

Assistant 2:

{prior_info2}

Your news sources say:
{summary_report}


IMPORTANT: The last thing you write is your final answer to the original question (XX) EXACTLY as:

"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"

Do not put "%" or any other unit after the numbers! Note that it is usually better to be careful with predictions.
This means that low percentiles should not be too high and high percentiles should not be too low compared to your median prediction.
Only choose very concentrated density functions if you are very sure.

"""

## MC

PROMPT_TEMPLATE_MC = """
IMPORTANT: Assume that today is {today}.

You are a professional forecaster. You read the question very carefully and understand it.
You are hard working and don't take short cuts. You follow all instructions precisely. Your answer can be long. Make sure to take as much space as you need.
I will mark some instructions as IMPORTANT. Make sure that you pay special attention here and follow closely, as you had trouble following these in the past!
Thank you!

First, you state the question, your options, background information, and resolution criteria.

Your question is:
{title}

With options:
{options}

Background:
{background}

Resolution criteria:
{resolution_criteria}

Make sure that you have properly understood the question, your options, background, and resolution criteria. You will answer a MULTIPLE CHOICE question, meaning
that you give all of your answers as probabilities that sum up to 100% as so:

Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N

IMPORTANT: Probabilities must always sum up to 100%!

IMPORTANT: Example of Resolution:
If there is a formula that is relevant for grading the question, state it explicitely. Provide an example for the individual factors making up the formula. Calculate how the question would be resolved.

You ALWAYS forecast questions according to the following procedure, you do not skip any step or substep:

1. State the current date, the date when the outcome to the question is known, and the number of days in between. Check whether there is reliable and current betting odds/metaculus probabilities on the question to form a prior. Take the betting odds and calculate an implied probability, while keeping in mind house edge.
If the question is an exact match (same question, same date, same resolution criteria, etc.) to the betting odds, do not change it further and just report it. In this case, you should be relatively certain about your estimate.
Otherwise, try to form a prior while considering the betting odds and continue with step 2.IMPORTANT: When considering predictions for different time frames, ALWAYS state the implied base rate on a monthly or yearly basis. Then form your prior based on that.
The closer the question can be matched to the betting odds question, the less should you change your opinion
with new information from steps 2+.

State your current estimate and your current certainty [0-10].

2.If there are no betting odds/metaculus probabilities, form a prior from your general knowledge of the world.
How strong is your prior? If your prior is strong, don't let the next steps influence you too much. IMPORTANT: If your assistants disagree, don't rely on their info too strongly!

Before stating your prior, first state:
(a) The number of days until the outcome is known.
(b) What the outcome would be if nothing changed.
(c) Consider whether it is reasonable that the outcome would change in the given timeframe. What is your prior given year on year base rates?
(d) If the question relates to a specific date, consider the cases that 1/4th of the time was left and that 4 times the time was left. What would your prior be then? Note that oftentimes more
time means more chances for something to happen. This may increase the probability for change.

State your current estimate and your current certainty [0-10].

3. Use the news output from your research assistant and update the prior with the new information, depending on how relevant and trustworthy the information is.
Consider whether sources may be biased. In general, there might be a slight pro left-wing bias in western sources and pro governemnt biases in general.

State your current estimate and your current certainty [0-10].

Fine print:
{fine_print}

Your Metaculus assistant says:
{meta_assistant}

Average predictions from Metaculus Quarterly Cup:

{predictions_full}

Further information that may help to update the prior from your assistants. IMPORTANT: Be careful with this information if the assistants disagree!

Assistant 1:

{prior_info}

Assistant 2:

{prior_info2}

Your news sources say:
{summary_report}


IMPORTANT: The last thing you write is your final answer to the original question EXACTLY as:

Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N

"""

## Prior prompt

PROMPT_PRIOR = """
IMPORTANT: Assume that today is {today}.

You are an assistant to a superforecaster, you do not produce predictions yourself. Be concise and precise. No hints or other commentary.
Construct a single prompt on the single most important topic that allows the superforecaster to search the web for information that will help him to forecast.
If the question relates to the decision making of some entity, always consider important dates regarding decision making.
It is better to include only the most relevant keywords in your prompt than too many unimportant ones. Keep it simple.
Consider whether info from the question may be outdated.


Your question is:
{title}

With options:
{options}

Background:
{background}

Resolution criteria:
{resolution_criteria}

Fine_print:
{fine_print}

Your answer should be the prompt, which will be delivered to a search engine. Do not include any further reasoning or anything else, only the prompt.

"""

## News aggregator

PROMPT_NEWS_AGG = """ You are an assistant to a superforecaster. The forecaster wants to make a prediction for the question:{title}

With options:
{options}

Background:
{background}

Resolution criteria:
{resolution_criteria}

Fine_print:
{fine_print}

Your job is to write a summary for the important pieces of news that relate to the question.
The news articles are:
{summary_report}

"""

## Fact checker

PROMPT_FACT_CHECKER = """
IMPORTANT: Assume that today is {today}.

A superforecaster has made a prediction for the question: {title}

Your job is to critically evaluate the forecaster's prediction.

1. State: the current date, the date when the outcome to the question is known, and the number of days in between.

2. If the forecaster used Metaculus predictions to inform his prior, did he properly account for differences between the question at hand and the predictions? Did he consider proper base rates?
IMPORTANT: If he used Metaculus predictions and they were highly relevant to the question, the prediction of the forecaster should not differ much from the Metaculus predictions!

3. Is the forecaster's logic sound?

4. Did he use all important information?

5. Did he properly assess the importance of different pieces of information?

6. Did he properly consider the available timeframe until the question is resolved? Pay special attention to this. Many questions have a short timeframe, which strongly limits the probability
that the question resolves positively. Reason through this in detail. IMPORTANT: If the question asks about an event that has already resolved but you don't know how, a reasonable estimate should
nevertheless be provided. Do not answer 0% in this case!

7. Are the predictions consistent with the predictions for the other questions?

8. Did the assisstants agree or disagree? Did the forecaster handle major disagreement with care by updating less strongly on the assistants' information?

Here is the forecaster's rationale:

{rationale}

Here is more information that the forecaster had available.

Background:
{background}

Resolution criteria:
{resolution_criteria}

Fine_print:
{fine_print}


Your Metaculus assistant says:
{meta_assistant}

Average predictions from Metaculus Quarterly Cup:

{predictions_full}

Further information that may help to update the prior from your assistants. IMPORTANT: Be careful with this information if the assistants disagree!

Assistant 1:

{prior_info}

Assistant 2:

{prior_info2}

Your news sources say:
{summary_report}

Critically evaluate the forecaster's prediction. IMPORTANT: Be sure to stay consistent with old predictions for related questions!
The sum of questions that are both mututally exclusive and exhaustive must be 100%!

At the end, do the following without further commentary!

If you see no room for improvement, state the forecaster's prediction EXACTLY as: "Probability: ZZ%", ZZ in 0-100, integer (NO COMMA OR DOT), IMPORTANT: DO NOT FORGET THE % SIGN OR THE PREDICTION WON'T COUNT!.

If you see room for improvement, state your updated prediction EXACTLY as: "Probability: ZZ%", ZZ in 0-100, integer (NO COMMA OR DOT), IMPORTANT: DO NOT FORGET THE % SIGN OR THE PREDICTION WON'T COUNT!.

"""

## Fact checker

PROMPT_FACT_CHECKER_NUMERIC = """
IMPORTANT: Assume that today is {today}.

A superforecaster has made a prediction for the question: {title}

Your job is to critically evaluate the forecaster's prediction.

1. State: the current date, the date when the outcome to the question is known, and the number of days in between.

2. If the forecaster used Metaculus predictions to inform his prior, did he properly account for differences between the question at hand and the predictions? Did he consider proper base rates?
IMPORTANT: If he used Metaculus predictions and they were highly relevant to the question, the prediction of the forecaster should not differ much from the Metaculus predictions!

3. Is the forecaster's logic sound?

4. Did he use all important information?

5. Did he properly assess the importance of different pieces of information?

6. Did he properly consider the available timeframe until the question is resolved? Pay special attention to this. Many questions have a short timeframe, which strongly limits the probability
that things change from their current situation! Reason through this in detail. IMPORTANT: If the question asks about an event that has already resolved but you don't know how, a reasonable estimate should
nevertheless be provided. Do not answer 0% in this case!

7. Did the assisstants agree or disagree? Did the forecaster handle major disagreement with care by updating less strongly on the assistants' information?

Here is the forecaster's rationale:

{rationale}

Here is more information that the forecaster had available.

Background:
{background}

Resolution criteria:
{resolution_criteria}

Fine_print:
{fine_print}

Your Metaculus assistant says:
{meta_assistant}

Average predictions from Metaculus Quarterly Cup:

{predictions_full}

Further information that may help to update the prior from your assistants. IMPORTANT: Be careful with this information if the assistants disagree!

Assistant 1:

{prior_info}

Assistant 2:

{prior_info2}

Your news sources say:
{summary_report}

Critically evaluate the forecaster's prediction.

At the end, do the following without further commentary! Do not put "%" after the numbers!

If you see no room for improvement, state the forecaster's prediction (XX) EXACTLY as:

"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"

If you see room for improvement, state your updated prediction (XX) EXACTLY as:

"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"

Note that it is usually better to be careful with predictions.
This means that low percentiles should not be too high and high percentiles should not be too low compared to your median prediction.
Only choose very concentrated density functions if you are very sure.

"""

## Fact checker

PROMPT_FACT_CHECKER_MC = """
IMPORTANT: Assume that today is {today}.

A superforecaster has made a prediction for the question: {title}

Your job is to critically evaluate the forecaster's prediction.

1. State: the current date, the date when the outcome to the question is known, and the number of days in between.

2. If the forecaster used Metaculus predictions to inform his prior, did he properly account for differences between the question at hand and the predictions? Did he consider proper base rates?
IMPORTANT: If he used Metaculus predictions and they were highly relevant to the question, the prediction of the forecaster should not differ much from the Metaculus predictions!

3. Is the forecaster's logic sound?

4. Did he use all important information?

5. Did he properly assess the importance of different pieces of information?

6. Did he properly consider the available timeframe until the question is resolved? Pay special attention to this. Many questions have a short timeframe, which strongly limits the probability
that things change from their current situation! Reason through this in detail. IMPORTANT: If the question asks about an event that has already resolved but you don't know how, a reasonable estimate should
nevertheless be provided. Do not answer 0% in this case!

7. Did the assisstants agree or disagree? Did the forecaster handle major disagreement with care by updating less strongly on the assistants' information?

8. IMPORTANT: Do the probabilities sum up to 100%? They must! Only provide probabilities that sum up to 100%!

Here is the forecaster's rationale:

{rationale}

Here is more information that the forecaster had available.

Background:
{background}

Resolution criteria:
{resolution_criteria}

Fine_print:
{fine_print}

Your Metaculus assistant says:
{meta_assistant}

Average predictions from Metaculus Quarterly Cup:

{predictions_full}

Further information that may help to update the prior from your assistants. IMPORTANT: Be careful with this information if the assistants disagree!

Assistant 1:

{prior_info}

Assistant 2:

{prior_info2}

Your news sources say:
{summary_report}

Critically evaluate the forecaster's prediction.

At the end, do the following without further commentary!

If you see no room for improvement, state the forecaster's prediction in this order {options} EXACTLY as:

Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N

If you see room for improvement, state your updated prediction in this order {options} EXACTLY as:

Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N

"""
## META checker

PROMPT_META_CHECKER = """
Assume that today is {today}. You are an assistant to a forecaster.

Today's questions are: {predictions_section}

Open questions on Metaculus are: {meta_open}

Your job is to report the id's of Metaculus questions that are related to today's questions. Report them as as a list with "ID:" in front of every id. Example: "ID: 23412 ID: 34555"
IMPORTANT: Do not report anything else! Do not include reasoning! Your response will be directly used in the pipeline.

"""

#CELL 2

AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"
API_BASE_URL2 = "https://www.metaculus.com/api2"

def post_question_comment(post_id: int, comment_text: str) -> None:
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS,
    )
    if not response.ok:
        raise Exception(response.text)


def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a forecast on a question.
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    response = requests.post(
        url,
        json=[
            {
                "question": question_id,
                **forecast_payload,
            },
        ],
        **AUTH_HEADERS,
    )
    print(response)
    if not response.ok:
        raise Exception(response.text)

def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a dictionary that maps
      quartiles or percentiles to datetimes, or a 201 value cdf.
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }

def get_question_details(question_id: int) -> dict:
    """
    Get all details about a question post.
    """
    url = f"{API_BASE_URL}/questions/{question_id}/"
    print(url)
    response = requests.get(
        url,
        **AUTH_HEADERS,
    )
    if not response.ok:
        raise Exception(response.text)
    return json.loads(response.content)


def list_posts(tournament_id, offset=0, count=50) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join([
            "binary",
            "multiple_choice",
            "numeric",
        ]),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data

def get_asknews_context(query: str) -> tuple[str, str]:
    """
    Use the AskNews `news` endpoint to get news context for your query.
    The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
    """
    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID,
        client_secret=ASKNEWS_SECRET,
        scopes=["news"]
    )

    # get the latest news related to the query (within the past 48 hours)
    hot_response = ask.news.search_news(
        query=query, # your natural language query
        n_articles=10, # control the number of articles to include in the context, originally 5
        return_type="both",
        strategy="latest news" # enforces looking at the latest news only
    )

    # get context from the "historical" database that contains a news archive going back to 2023
    historical_response = ask.news.search_news(
        query=query,
        n_articles=10,
        return_type="both",
        strategy="news knowledge" # looks for relevant news within the past 60 days
    )

    # you can also specify a time range for your historical search if you want to
    # slice your search up periodically.
    # now = datetime.datetime.now().timestamp()
    # start = (datetime.datetime.now() - datetime.timedelta(days=100)).timestamp()
    # historical_response = ask.news.search_news(
    #     query=query,
    #     n_articles=20,
    #     return_type="both",
    #     historical=True,
    #     start_timestamp=int(start),
    #     end_timestamp=int(now)
    # )

    news_articles_with_full_context = hot_response.as_string + historical_response.as_string
    formatted_articles = format_asknews_context(
        hot_response.as_dicts, historical_response.as_dicts)
    return news_articles_with_full_context, formatted_articles


def format_asknews_context(hot_articles: list[dict], historical_articles: list[dict]) -> str:
    """
    Format the articles for posting to Metaculus.
    """

    formatted_articles = "Here are the relevant news articles:\n\n"

    if hot_articles:
      hot_articles = [article.__dict__ for article in hot_articles]
      hot_articles = sorted(
          hot_articles, key=lambda x: x['pub_date'], reverse=True)

      for article in hot_articles:
          pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
          formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if historical_articles:
      historical_articles = [article.__dict__ for article in historical_articles]
      historical_articles = sorted(
          historical_articles, key=lambda x: x['pub_date'], reverse=True)

      for article in historical_articles:
          pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
          formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if not hot_articles and not historical_articles:
      formatted_articles += "No articles were found.\n\n"
      return formatted_articles

    # formatted_articles += f"*Generated by AI at [AskNews](https://asknews.app), check out the [API](https://docs.asknews.app) for more information*."

    return formatted_articles

def extract_prediction_from_response_as_percentage_not_decimal(forecast_text: str) -> float:
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        # Return the last number found before a '%'
        number = int(matches[-1])
        number = min(99, max(1, number))  # clamp the number between 1 and 99
        return number
    else:
        raise ValueError(
            f"Could not extract prediction from response: {forecast_text}"
        )

def extract_meta_id(forecast_text: str) -> float:
  matches = re.findall(r"(\d+)", forecast_text)
  if matches:
      # Return the last number
      number = int(matches[-1])
      return number
  else:
      raise ValueError(
          f"Could not extract prediction from response: {forecast_text}"
      )


def get_gpt_prediction(question_details: dict,question_id, num_runs: int = 1) -> tuple[float, str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]

    if question_type == "multiple_choice":
      options = question_details["options"]
    else:
      options = "No options available"
    if question_type == "numeric":
      scaling = question_details["scaling"]
      open_upper_bound = question_details["open_upper_bound"]
      open_lower_bound = question_details["open_lower_bound"]
      upper_bound = scaling["range_max"]
      lower_bound = scaling["range_min"]
    else:
      open_upper_bound=True
      open_lower_bound=True
    # Create messages about the bounds that are passed in the LLM prompt
    if open_upper_bound:
      upper_bound_message = ""
    else:
      upper_bound_message = f"The outcome can not be higher than {upper_bound}."
    if open_lower_bound:
      lower_bound_message = ""
    else:
      lower_bound_message = f"The outcome can not be lower than {lower_bound}."

    if GET_NEWS == True:
      # If you want to use AskNews, use the below
      full_article_context, formatted_articles = get_asknews_context(title)
      summary_report = formatted_articles

      # If you want to use Perplexity, use the below
      # summary_report += call_perplexity(title)

      content = PROMPT_NEWS_AGG.format(
      title=title,
      today=today,
      background=background,
      resolution_criteria=resolution_criteria,
      fine_print=fine_print,
      summary_report=summary_report,
      options=options
  )
      url = "https://www.metaculus.com/proxy/anthropic/v1/messages/"
      headers = {
          "Authorization": f"Token {METACULUS_TOKEN}",
          "anthropic-version": "2023-06-01",
          "Content-Type": "application/json"
      }
      json_code = {
          "model": "claude-3-5-sonnet-20241022",
          "max_tokens": 4096,
          "temperature": 0.1,
          "messages": [
              {
                  "role": "user",
                  "content": content
              }
          ]
      }

      response = requests.post(url, headers=headers,json=json_code)
      response.raise_for_status()

      response_data = response.json()
      summary_report_agg = response_data['content'][0]['text']

    else:
      summary_report = ""

    # Call betting lines generator

  #  rationale = get_betting_lines(question_details)

  # Call prior generator

    prior_prompt = get_prior(question_details)

    assistant_prompt_prior = f"""
Assume that today is {today}. You are an assistant to a superforecaster.
You will receive a prompt to search the web.
"""


    query_prior = f"""This is what the superforecaster asks of you: {prior_prompt}.
    For context, his question is: {title}
    Background:
    {background}
    Resolution criteria:
    {resolution_criteria}
    Fine_print:
    {fine_print}"""

    messages_prior = [
    {
        "role": "system",
        "content": assistant_prompt_prior,
    },
    {
        "role": "user",
        "content": query_prior,
    },
    ]

    # Call Perplexity for prior info

    prior_info = call_perplexity_with_messages(messages_prior)

    # CALL PERPLEXITY AGAIN!

    prior_prompt2 = get_prior(question_details)

    assistant_prompt_prior = f"""
Assume that today is {today}. You are an assistant to a superforecaster.
You will receive a prompt to search the web.
"""


    query_prior = f"""This is what the superforecaster asks of you: {prior_prompt2}.
    For context, his question is: {title}
    Background:
    {background}
    Resolution criteria:
    {resolution_criteria}
    Fine_print:
    {fine_print}"""

    messages_prior = [
    {
        "role": "system",
        "content": assistant_prompt_prior,
    },
    {
        "role": "user",
        "content": query_prior,
    },
    ]

    # Call Perplexity for prior info

    prior_info2 = call_perplexity_with_messages(messages_prior)

    # Call Perplexity for Metaculus predictions
    assistant_prompt_meta2 = """
You are an assistant to a superforecaster. You will provide the Metaculus ID to the question in the prompt. If you did not find a precise match but you did find a similar question, also return an ID without comment.
If you found something, your answer consists of just the ID. If you didn't find anything, your answer consists only of the number 0. Please do not write anything else under any circumstance as it will destroy the prediction pipeline.
IMPORTANT: YOUR LAST WORD MUST BE THE ID AND NOTHING ELSE. THIS IS VERY IMPORTANT!"""
    query_meta = f"Search metaculus for questions similar to this one: {title}"
    messages_meta = [
        {
            "role": "system",
            "content": assistant_prompt_meta2,
        },
        {
            "role": "user",
            "content": query_meta,
        },
        ]

    meta_info2 = call_perplexity_with_messages(messages_meta)
    meta_info2 =extract_meta_id(meta_info2)
    try:
        question_id = int(meta_info2)  # Convert to integer if meta_info2 is a valid ID
    except ValueError:
        print("Invalid question ID received from meta_info2.")
        print(question_id)
        question_id = 0
    if question_id:
        url = f"https://www.metaculus.com/api/posts/{question_id}/"
        response = requests.get(url,headers={"Authorization": f"Token {METACULUS_TOKEN}"})
        data = response.json()
        if data.get('question', {}).get('type', {}) == "binary": # BINARY
          prediction = get_community_prediction(question_id)
          if prediction:
            meta_title=f"Question Title: {prediction['title']}"
            meta_mean=f"Mean Prediction: {prediction['mean']}"
            meta_time=f"{prediction['resolution_date']}"
          else:
            meta_title=0
            meta_mean=0
            meta_time=0
        elif data.get('question', {}).get('type', {}) == "numeric": # NUMERIC
          prediction = extract_numeric_prediction(data)
          if prediction:
            meta_title=f"Question Title: {prediction['title']}"
            meta_mean=f"Mean Prediction: {prediction['prediction']}, Upper quartile: {prediction['upper_quartile']}, Lower quartile: {prediction['lower_quartile']}"
            meta_time=f"{prediction['resolution_date']}"
          else:
            meta_title=0
            meta_mean=0
            meta_time=0
        else:
            meta_title=0
            meta_mean=0
            meta_time=0
    else:
        meta_title=0
        meta_mean=0
        meta_time=0


    print(f"\n\n--------META ID AND MEAN----------")
    print(meta_title)
    print(meta_mean)
    print(meta_time)
    print(title)
    print(f"\n\n----END META ID----")

# Check whether betting lines are proper

    if meta_title:
        resolution_date = datetime.datetime.fromisoformat(meta_time.replace('Z', '+00:00'))
        today_naive = datetime.datetime.strptime(today, "%Y-%m-%d")
        today_aware = today_naive.replace(tzinfo=datetime.timezone.utc)

        if today_aware < resolution_date:
          meta_assistant=f"Metaculus probabilities for the question {meta_title} to resolve positively are {meta_mean}. "
        else:
          meta_assistant="I did not find Metaculus predictions"


    else:
        meta_assistant="I did not find Metaculus predictions"

    # print(f"\n\n--------META ASSISTANT----------")
    # print(meta_assistant)
    # print(f"\n\n----END LLM PROMPT----")

    # Predictions from Metaculus API

    # binary
    url = "https://www.metaculus.com/api/posts/?tournaments=quarterly-cup&statuses=open&forecast_type=binary"
    response = requests.get(url)
    data = response.json()
    questions_coarse = data['results']
    question_ids_binary = [q['id'] for q in questions_coarse]

    predictions_binary = []
    for q_id in question_ids_binary:
        prediction = get_community_prediction(q_id)
        if prediction:
          meta_title=f"Question Title: {prediction['title']}"
          meta_mean=f"Mean Prediction: {prediction['mean']}"
          predictions_binary.append(meta_title)
          predictions_binary.append(meta_mean)

    # numeric
    url = "https://www.metaculus.com/api/posts/?tournaments=quarterly-cup&statuses=open&forecast_type=numeric"
    response = requests.get(url)
    data = response.json()
    questions_coarse = data['results']
    question_ids_numeric = [q['id'] for q in questions_coarse]


    predictions_numeric = []
    for q_id in question_ids_numeric:
        url = f"https://www.metaculus.com/api/posts/{q_id}/"
        response = requests.get(url)
        data = response.json()
        prediction = extract_numeric_prediction(data)
        predictions_numeric.append(prediction)

    predictions_full = predictions_binary
    predictions_full.extend(predictions_numeric)

    ## Payload for Forecaster AI

    if question_type == "binary":
      content = PROMPT_TEMPLATE.format(
          title=title,
          today=today,
          background=background,
          resolution_criteria=resolution_criteria,
          fine_print=fine_print,
          summary_report=summary_report_agg,
          meta_assistant=meta_assistant,
          prior_info=prior_info,
          prior_info2=prior_info2,
          predictions_full=predictions_full,
      )
    if question_type == "multiple_choice":
      content = PROMPT_TEMPLATE_MC.format(
          title=title,
          today=today,
          background=background,
          resolution_criteria=resolution_criteria,
          fine_print=fine_print,
          summary_report=summary_report_agg,
          meta_assistant=meta_assistant,
          prior_info=prior_info,
          prior_info2=prior_info2,
          predictions_full=predictions_full,
          options=options
      )
    if question_type == "numeric":
      content = PROMPT_TEMPLATE_NUMERIC.format(
          title=title,
          today=today,
          background=background,
          resolution_criteria=resolution_criteria,
          fine_print=fine_print,
          summary_report=summary_report_agg,
          meta_assistant=meta_assistant,
          prior_info=prior_info,
          prior_info2=prior_info2,
          predictions_full=predictions_full,
          lower_bound_message=lower_bound_message,
          upper_bound_message=upper_bound_message,
          scaling_min=lower_bound,
          scaling_max=upper_bound
      )


    PRINT_LLM_PROMPT = True
    if PRINT_LLM_PROMPT:
        print(f"\n\n--------LLM PROMPT----------")
        print(content)
        print(f"\n\n----END LLM PROMPT----")

    probabilities = []
    rationales = []

    for i in range(num_runs):
        # Retry logic
        max_retries = 10
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                url = "https://www.metaculus.com/proxy/anthropic/v1/messages/"

                headers = {
                    "Authorization": f"Token {METACULUS_TOKEN}",
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }

                json_code = {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 4096,
                    "temperature": 0.1,
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                }
                response = requests.post(url, headers=headers,json=json_code)
                response.raise_for_status()
                response_data = response.json()
                rationale = response_data['content'][0]['text']

                # Fact Checker
                if question_type == "binary":
                  content_fact_checker = PROMPT_FACT_CHECKER.format(
                        title=title,
                        today=today,
                        background=background,
                        resolution_criteria=resolution_criteria,
                        fine_print=fine_print,
                        summary_report=summary_report_agg,
                        meta_assistant=meta_assistant,
                        prior_info=prior_info,
                        prior_info2=prior_info2,
                        rationale=rationale,
                        predictions_full=predictions_full
                    )
                if question_type == "numeric":
                  content_fact_checker = PROMPT_FACT_CHECKER_NUMERIC.format(
                        title=title,
                        today=today,
                        background=background,
                        resolution_criteria=resolution_criteria,
                        fine_print=fine_print,
                        summary_report=summary_report_agg,
                        meta_assistant=meta_assistant,
                        prior_info=prior_info,
                        prior_info2=prior_info2,
                        rationale=rationale,
                        predictions_full=predictions_full,
                        lower_bound_message=lower_bound_message,
                        upper_bound_message=upper_bound_message
                    )
                if question_type == "multiple_choice":
                  content_fact_checker = PROMPT_FACT_CHECKER_MC.format(
                        title=title,
                        today=today,
                        background=background,
                        resolution_criteria=resolution_criteria,
                        fine_print=fine_print,
                        summary_report=summary_report_agg,
                        meta_assistant=meta_assistant,
                        prior_info=prior_info,
                        prior_info2=prior_info2,
                        rationale=rationale,
                        predictions_full=predictions_full,
                        options=options
                    )
                json_code = {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 4096,
                    "temperature": 0.1,
                    "messages": [
                        {
                            "role": "user",
                            "content": content_fact_checker
                        }
                    ]
                }
                response = requests.post(url, headers=headers,json=json_code)
                response.raise_for_status()
                response_data = response.json()
                rationale2 = response_data['content'][0]['text']
                break
            except requests.exceptions.RequestException as e:
                print(f"Request failed (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise
        if question_type == "binary":
          probability = extract_prediction_from_response_as_percentage_not_decimal(rationale2)
          probabilities.append(probability)
        if question_type == "multiple_choice":
          option_probabilities = (
          extract_option_probabilities_from_response(rationale2, options)
          )
          probabilities.append(option_probabilities)
        if question_type == "numeric":
          percentile_values = (extract_percentiles_from_response(rationale))
          probabilities.append(percentile_values)
        rationales.append(rationale)
        rationales.append(rationale2)

    if question_type == "binary":
      avg_probability = sum(probabilities) / len(probabilities)
      # Prepare the comment to post
      comment = f"Average Probability: {avg_probability:.2f}%\n\nIndividual Probabilities: {probabilities}\n\nClaude's Answers:\n"
      for idx, rationale in enumerate(rationales):
          comment += f"Run {idx+1}:\n{rationale}\n\n"
      avg_probability=avg_probability/100
      return avg_probability, comment

    if question_type == "multiple_choice":
      num_options = len(probabilities[0])  # Assuming all sub-lists have the same length
      option_probabilities = [sum(sublist[i] for sublist in probabilities) / len(probabilities) for i in range(num_options)]
      comment = f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nClaude's Answers:\n"
      for idx, rationale in enumerate(rationales):
          comment += f"Run {idx+1}:\n{rationale}\n\n"
      probability_yes_per_category = generate_multiple_choice_forecast(options, option_probabilities)
      return probability_yes_per_category, comment

    if question_type == "numeric":
      percentile_values = {}
      for percentile in probabilities[0].keys():  # Assuming all dictionaries have the same percentiles
            values = [d.get(percentile, 0) for d in probabilities]  # Get values for the current percentile from all dictionaries
            percentile_values[percentile] = sum(values) / len(values)  # Calculate average
      comment = f"Extracted Percentile_values: {percentile_values}%\n\nClaude's Answers:\n"
      for idx, rationale in enumerate(rationales):
          comment += f"Run {idx+1}:\n{rationale}\n\n"
      print(f"Extracted Percentile_values: {percentile_values}")
      print(f"Scaling: {scaling}")
      print(f"Open upper bound: {open_upper_bound}")
      print(f"Open lower bound: {open_lower_bound}")

      cdf = generate_continuous_cdf(percentile_values, question_type, open_upper_bound, open_lower_bound, scaling)

      return cdf, comment


# Updated function to match your previous Perplexity setup
def call_perplexity_with_messages(messages: list) -> str:
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    if not PERPLEXITY_API_KEY:
        print("PERPLEXITY_API_KEY is not set.")
        return "No information found."
    url = "https://api.perplexity.ai/chat/completions"
    api_key = PERPLEXITY_API_KEY
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    payload = {
        "model": "llama-3.1-sonar-huge-128k-online",
        "messages": messages,
    }
    response = requests.post(url=url, json=payload, headers=headers)
    if not response.ok:
        print("Error fetching data from Perplexity:", response.text)
        return "No information found."
    content = response.json()["choices"][0]["message"]["content"]
    return content


def get_prior(question_details: dict) -> str:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    if question_details["type"] == "multiple_choice":
      options = question_details["options"]
    else:
      options = "No options available"

    content = PROMPT_PRIOR.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        options=options
    )

    PRINT_PRIOR_PROMPT = False
    if PRINT_PRIOR_PROMPT:
        print(f"\n\n--------PRIOR PROMPT----------")
        print(content)
        print(f"\n\n----END PRIOR PROMPT----")

    url = "https://www.metaculus.com/proxy/anthropic/v1/messages/"

    headers = {
        "Authorization": f"Token {METACULUS_TOKEN}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }

    json_code = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }
    response = requests.post(url, headers=headers,json=json_code)
    print(response)
    response.raise_for_status()
    response_data = response.json()
    rationale = response_data['content'][0]['text']
    print(rationale)
    return rationale

def get_community_prediction(question_id):
    """Get latest community prediction for a Metaculus question"""
    url = f"https://www.metaculus.com/api2/questions/{question_id}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    # Get the prediction history
    history = data.get('question', {}).get('aggregations', {}).get('recency_weighted', {}).get('history', [])
    title = data.get('question', {}).get('title')

    # Get resolution time - try scheduled first, then actual if available
    resolution_time = data.get('question', {}).get('scheduled_resolve_time')
    if not resolution_time:
        resolution_time = data.get('question', {}).get('actual_resolve_time')

    # Convert ISO format date to more readable format if it exists
    if resolution_time:
        try:
            resolution_date = datetime.fromisoformat(resolution_time.replace('Z', '+00:00'))
            resolution_time = resolution_date.strftime('%Y-%m-%d')
        except:
            pass  # Keep original format if conversion fails

    if history:
        # Get the most recent prediction (last item in history)
        latest = history[-1]
        return {
            'title': title,
            'mean': latest.get('means', [None])[0],
            'forecaster_count': latest.get('forecaster_count'),
            'resolution_date': resolution_time
        }
    return None
# Extract links from resolution criteria
def extract_links(resolution_criteria):
    # This pattern matches URLs within markdown links [text](url) and bare URLs
    pattern = r'\[.*?\]\((.*?)\)|(?:https?://\S+)'

    # Find all matches
    links = re.findall(pattern, resolution_criteria)

    # Clean up the results (remove any empty strings and clean any remaining parentheses)
    links = [link.strip('()') for link in links if link]

    return links

def search_resolution_links(resolution_criteria):
    # Extract links
    pattern = r'\[.*?\]\((.*?)\)|(?:https?://\S+)'
    links = re.findall(pattern, resolution_criteria)
    links = [link.strip('()') for link in links if link]

    # Only proceed if links were found
    if not links:
        return None

    # Search each link with asknews
    search_results = []
    for link in links:
        try:
            messages_prior = [
           {
             "role": "system",
             "content": "Access the following link and provide a summary of its contents",
           },
           {
             "role": "user",
             "content": link,
           },
    ]

    # Call Perplexity for prior info

            prior_info = call_perplexity_with_messages(messages_prior)
            search_results.append(result)
        except Exception as e:
            print(f"Error searching link {link}: {e}")

    return search_results


def extract_numeric_prediction(data):
    # Title
    title = data['title']
    # Get scaling parameters
    range_max = data['question']['scaling']['range_max']
    range_min = data['question']['scaling']['range_min']

    # Get resolution time - try scheduled first, then actual if available
    resolution_time = data.get('question', {}).get('scheduled_resolve_time')
    if not resolution_time:
        resolution_time = data.get('question', {}).get('actual_resolve_time')

    # Convert ISO format date to more readable format if it exists
    if resolution_time:
        try:
            resolution_date = datetime.fromisoformat(resolution_time.replace('Z', '+00:00'))
            resolution_time = resolution_date.strftime('%Y-%m-%d')
        except:
            pass  # Keep original format if conversion fails

    # Get latest prediction
    history = data['question']['aggregations']['recency_weighted']['history']
    if history:
        latest = history[-1]

        # Get the center value from the latest prediction
        scaled_center = latest['centers'][0]  # This is in [0,1] scale

        # Convert from [0,1] scale to actual value using Metaculus linear scaling
        actual_value = range_min + (scaled_center * (range_max - range_min))

        # Get confidence intervals if available
        scaled_lower = latest['interval_lower_bounds'][0]
        scaled_upper = latest['interval_upper_bounds'][0]

        actual_lower = range_min + (scaled_lower * (range_max - range_min))
        actual_upper = range_min + (scaled_upper * (range_max - range_min))


        return {
            'title': title,
            'prediction': round(actual_value, 2),
            'lower_quartile': round(actual_lower, 2),
            'upper_quartile': round(actual_upper, 2),
            'resolution_date': resolution_time
        }
    else:
        return {
            'title': title,
            'prediction': "None",
            'lower_quartile': "None",
            'upper_quartile': "None",
            'resolution_date': resolution_time
        }

def extract_percentiles_from_response(forecast_text: str) -> float:
    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_percentile_numbers(text):
        # Regular expression pattern
        pattern = r'^.*(?:P|p)ercentile.*$'

        # Number extraction pattern
        number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'

        results = []

        # Iterate through each line in the text
        for line in text.split('\n'):
            # Check if the line contains "Percentile" or "percentile"
            if re.match(pattern, line):
                # Extract all numbers from the line
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [num.replace(',', '') for num in numbers]
                # Convert strings to float or int
                numbers = [float(num) if '.' in num else int(num) for num in numbers_no_commas]
                # Add the tuple of numbers to results
                if len(numbers) > 1:
                  first_number = numbers[0]
                  last_number = numbers[-1]
                  tup = [first_number, last_number]
                  results.append(tuple(tup))

        # Convert results to dictionary
        percentile_values = {}
        for first_num, second_num in results:
            key = first_num
            percentile_values[key] = second_num

        return percentile_values

    percentile_values = extract_percentile_numbers(forecast_text)

    if len(percentile_values) > 0:
        return percentile_values
    else:
        raise ValueError(
            f"Could not extract prediction from response: {forecast_text}"
        )


def generate_continuous_cdf(
    percentile_values: dict,
    question_type: str,
    open_upper_bound: bool,
    open_lower_bound: bool,
    scaling: dict,
) -> list[float]:
    """
    Returns: list[float]: A list of 201 float values representing the CDF.
    """

    percentile_max = max(float(key) for key in percentile_values.keys())
    percentile_min = min(float(key) for key in percentile_values.keys())
    range_min = float(scaling.get("range_min"))
    range_max = float(scaling.get("range_max"))

    # Set cdf values outside range
    if open_upper_bound:
        if range_max > percentile_values[percentile_max]:
           percentile_values[int(100 - (0.5 * (100 - percentile_max)))] =  range_max
    else:
        percentile_values[100] = range_max

    # Set cdf values outside range
    if open_lower_bound:
        if range_min < percentile_values[percentile_min]:
           percentile_values[int(0.5 * percentile_min)] =  range_min
    else:
        percentile_values[0] = range_min

    print(f'Percentile_values {percentile_values.items()}')

    sorted_percentile_values = dict(sorted(percentile_values.items()))

    # Normalize percentile keys
    normalized_percentile_values = {}
    for key, value in sorted_percentile_values.items():
        percentile = float(key) / 100
        normalized_percentile_values[percentile] = value

    print(f'normalized_percentile_values: {normalized_percentile_values}')

    value_percentiles = {value: key for key, value in normalized_percentile_values.items()}

    print(f'value_percentiles: {value_percentiles}')


    range_min = scaling.get("range_min")
    range_max = scaling.get("range_max")
    zero_point = scaling.get("zero_point")


    # function for log scaled questions
    def generate_cdf_locations(range_min, range_max, zero_point):
        if zero_point is None:
            scale = lambda x: range_min + (range_max - range_min) * x
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scale = lambda x: range_min + (range_max - range_min) * (
                deriv_ratio**x - 1
            ) / (deriv_ratio - 1)
        return [scale(x) for x in np.linspace(0, 1, 201)]

    cdf_xaxis = generate_cdf_locations(range_min, range_max, zero_point)

    print(f'range_min: {range_min}')
    print(f'range_max: {range_max}')
    print(f'zero_point: {zero_point}')
    print(f'cdf_axis: {cdf_xaxis}\n')


    def linear_interpolation(x_values, xy_pairs):
        # Sort the xy_pairs by x-values
        sorted_pairs = sorted(xy_pairs.items())

        # Extract sorted x and y values
        known_x = [pair[0] for pair in sorted_pairs]
        known_y = [pair[1] for pair in sorted_pairs]

        # Initialize the result list
        y_values = []

        for x in x_values:
            # Check if x is exactly in the known x values
            if x in known_x:
                y_values.append(known_y[known_x.index(x)])
            else:
                # Find the indices of the two nearest known x-values
                i = 0
                while i < len(known_x) and known_x[i] < x:
                    i += 1

                list_index_2 = i

                # If x is outside the range of known x-values, use the nearest endpoint
                if i == 0:
                    y_values.append(known_y[0])
                elif i == len(known_x):
                    y_values.append(known_y[-1])
                else:
                    # Perform linear interpolation
                    x0, x1 = known_x[i-1], known_x[i]
                    y0, y1 = known_y[i-1], known_y[i]

                    # Linear interpolation formula
                    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                    y_values.append(y)

        return y_values

    continuous_cdf = linear_interpolation(cdf_xaxis, value_percentiles)


    print(f'continuous_cdf: {continuous_cdf}')

    return continuous_cdf
def extract_option_probabilities_from_response(forecast_text: str, options) -> float:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_option_probabilities(text):

        # Number extraction pattern
        number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'

        results = []

        # Iterate through each line in the text
        for line in text.split('\n'):
            # Extract all numbers from the line
            numbers = re.findall(number_pattern, line)
            numbers_no_commas = [num.replace(',', '') for num in numbers]
            # Convert strings to float or int
            numbers = [float(num) if '.' in num else int(num) for num in numbers_no_commas]
            # Add the tuple of numbers to results
            if len(numbers) >= 1:
              last_number = numbers[-1]
              results.append(last_number)

        return results

    option_probabilities = extract_option_probabilities(forecast_text)

    NUM_OPTIONS = len(options)

    if len(option_probabilities) > 0:
        # return the last NUM_OPTIONS items
        return option_probabilities[-NUM_OPTIONS:]
    else:
        raise ValueError(
            f"Could not extract prediction from response: {forecast_text}"
        )


def generate_multiple_choice_forecast(options, option_probabilities) -> dict:
    """
    Returns: dict corresponding to the probabilities of each option.
    """
    print(f'options: {options}')
    print(f'option_probabilities: {option_probabilities}')

    # confirm that there is a probability for each option
    if len(options) != len(option_probabilities):
      raise ValueError(f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})")

    # Ensure we are using decimals
    total_sum = sum(option_probabilities)
    decimal_list = [x / total_sum for x in option_probabilities]

    def normalize_list(float_list):
        # Step 1: Clamp values
        clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]

        # Step 2: Calculate the sum of all elements
        total_sum = sum(clamped_list)

        # Step 3: Normalize the list so that all elements add up to 1
        normalized_list = [x / total_sum for x in clamped_list]

        # Step 4: Adjust for any small floating-point errors
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment

        return normalized_list

    normalized_option_probabilities = normalize_list(decimal_list)

    probability_yes_per_category = {}
    for i in range(len(options)):
        probability_yes_per_category[options[i]] = normalized_option_probabilities[i]

    print(f'probability_yes_per_category: {probability_yes_per_category}')

    return probability_yes_per_category

def list_questions(tournament_id, offset=0, count=50) -> list[dict]:
    """
    List (all details) {count} questions from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "has_group": "false",
        "order_by": "-activity",
        "forecast_type": ",".join([
            "binary",
            "multiple_choice",
            "numeric",
        ]),
        "project": tournament_id,
        "status": "open",
        "type": "forecast",
        "include_description": "true",
    }
    url = f"{API_BASE_URL2}/questions/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data

# Cell 3
TOURNAMENT_ID = 32506  # 32506 is the tournament ID for Q4 AI Benchmarking
#TOURNAMENT_ID = 3672 # Quarterly Cup
#TOURNAMENT_ID = 2844 # ACX for testing

# @title Get all open questions from the tournament (TOURNAMENT_ID)

posts = list_posts(tournament_id=TOURNAMENT_ID)

print(posts["results"])

post_dict = dict()
for post in posts["results"]:
  print(f'question_id: {post["question"]["id"]} post_id: {post["id"]}.  \n')
  if question := post.get("question"):
      # single question post
      post_dict[post["id"]] = [question]

#print(f'post_dict: {post_dict}')

open_question_id_post_id = [] # [(question_id, post_id)]
new_question_id_post_id = []
for post_id, questions in post_dict.items():
    for question in questions:
        if question.get("status") == "open":
          print(
              f"ID: {question['id']}\nQ: {question['title']}\nCloses: "
              f"{question['scheduled_close_time']}"
          )
          open_question_id_post_id.append((question["id"], post_id))

questions=list_questions(TOURNAMENT_ID)
new_questions_ids = []
open_questions_ids = []
for question in questions["results"]:
    if question["status"] == "open":
        print(f"ID: {question['id']}\nQ: {question['title']}\nCloses: {question['scheduled_close_time']}")
        open_questions_ids.append(question["id"])

        # Check if you've made a prediction for this question
        BASE_URL = f"https://www.metaculus.com/api2"
        guess_response = requests.get(
            f"{BASE_URL}/questions/{question['id']}/",
            headers={"Authorization": f"Token {METACULUS_TOKEN}"}
        )
        guess_response.raise_for_status()

        if not guess_response.json().get("question", {}).get("my_forecasts", {}).get("latest"):
            new_questions_ids.append(question["id"])

print(f"New questions without predictions: {len(new_questions_ids)}")
print(new_questions_ids)

new_question_id_post_id = [
    entry for entry in open_question_id_post_id if entry[1] in new_questions_ids
]

print("Matching Entries:")
for entry in new_question_id_post_id:
    print(f"Question ID: {entry[0]}, Post ID: {entry[1]}")

print(f'open_question_id_post_id: {open_question_id_post_id}')


# Cell 4
# The list of questions to forecast
forecast_questions_ids = []
if FORECAST_TOURNAMENT == True:
    if ONLY_NEW:
      forecast_questions_ids = new_question_id_post_id
    else:
      forecast_questions_ids = open_question_id_post_id
else:
  forecast_questions_ids = [(30270, 30477)]
  # question_id: 30270 post_id: 30477 (Biden EO)
  # question_id: 30300 post_id: 30516 (Trump)
  # [(28571, 28571)] # (question_id, post_id)
  # [(28997, 29077)] brazil
  # (29480, 29608) elon
  # (28953, 29028) arms sales
  # (28571, 28571) SSE
  # (29051, 29141) Influenza A
  # (8529, 8529) Metaculus meetup
  # (29050, 29140) covid hospitalization

for question_id, post_id in forecast_questions_ids:

  question_details = get_question_details(question_id)
  title = question_details["title"]
  resolution_criteria = question_details["resolution_criteria"]
  background = question_details["description"]
  fine_print = question_details["fine_print"]
  question_type = question_details["type"]
  if question_type == "multiple_choice":
    options = question_details["options"]
    print(f"options: {options}")

  print(f"----------\nQuestion: {title}")

  forecast, comment = get_gpt_prediction(question_details,question_id, num_runs)

  print(f"Forecast: {forecast}")
  print(f"Comment: {comment}")

  forecast_payload = create_forecast_payload(forecast, question_type)
  post_question_prediction(question_details["id"], forecast_payload)
  post_question_comment(post_id, comment)
