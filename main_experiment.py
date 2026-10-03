import json
import pandas as pd
import random
import argparse
import asyncio
import datetime
from knockknock import slack_sender
from tqdm.asyncio import tqdm as asyncio_tqdm
from transformers import pipeline, AutoTokenizer
import logging
import torch
from openai import AsyncOpenAI
import tqdm
import copy
from huggingface_hub import login
import os
import time

with open("API_keys.json") as f:
    keys = json.load(f)
openAI_key = keys["openAI_key"]
hf_token = keys["HF_token"]

try:
    login(token=hf_token)
except Exception as e:
    print("ERROR HUGGING_FACE")
    print(e)
    pass

parser = argparse.ArgumentParser(description='Run news feed experiment')
parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                    help='the model to run the experiment on (default: meta-llama/Meta-Llama-3.1-8B-Instruct')
parser.add_argument('--temperature', type=float, default=1,
                    help='the model temperature (default: 1)')
parser.add_argument('--attribute', type=str, default='truth',
                    choices=['truth', 'interest', 'importance', 'sentiment'],
                    help='Attribute to rate statements on (default: truth)')

# parser.add_argument('--download_directory', type=str, default='ADD YOUR DOWNLOAD DIRECTORY FOR THE HR MODELS',
#                     help='the directory of downloading and loading HF models')

parser.add_argument('--download_directory', type=str, default='~/.cache/huggingface/hub',
                    help='the directory of downloading and loading HF models')

args = parser.parse_args()

nbr_simulations = 1
nbr_iterations = 5
feed_length = 4


# attribute = "truth"
# attribute = "importance"
# attribute = "sentiment"
# attribute = "interest"


# LLM_model = "gpt-5-nano"
# LLM_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# LLM_model = "Qwen/Qwen2.5-7B-Instruct"

# LLM_model = "google/gemma-3-4b-it"


# LLM_model = "mistralai/Mistral-7B-Instruct-v0.3" #=> doesn't get the change of the task during the rating phase
# LLM_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# LLM_model = "meta-llama/Llama-2-13b-chat-hf"

# LLM_model = "gpt-3.5-turbo-0125"
# LLM_model = "test"
# temperature = 0.7

LLM_model = args.model
temperature = args.temperature
attribute = args.attribute
CUSTOM_MODEL_CACHE_DIR = args.download_directory

nbr_trials = 1  # Single run is enough now => previously checked with three, no difference across trials

def get_daytime():
    now = datetime.datetime.now()
    minute = str(now.minute)
    hour = str(now.hour)
    day = str(now.day)
    month = str(now.month)
    if len(minute) == 1:
        minute = "0" + minute
    if len(hour) == 1:
        hour = "0" + hour
    if len(day) == 1:
        day = "0" + day
    if len(month) == 1:
        month = "0" + month
    return str(now.year) + "_" + month + "_" + day + "_" + hour + "_" + minute

def extract_actions(output_text, length_feed = 4):
    # ex output_text "T1: reshare \nT2: do nothing\nT3: do nothing\nT4: do nothing"
    # TODO This is still buggy for GPT-3.5-turbo-0125, mistral
    actions = []
    comments = []
    for i in range(1, length_feed + 1):
        try:
            action = output_text.split("T" + str(i) + ": ")[1].split("\n")[0]
        except:
            action = "non_valid"
        # may need to uncomment this for cleaner output
        # if "comment" in action:
        #     action = "comment"
        comment = ""
        if ("comment" in action.lower()) or ("reshare" in action.lower()):
            try:
                comment = action.split("comment with")[1].strip()
            except:
                comment = "non_valid"
        comments.append(comment)
        actions.append(action)
    return actions, comments


# initialize experiment system prompt
system_prompt = "You are a Twitter/X user. In the following, I will send you a list of tweets you would see when scrolling your news feed. For every tweet you see, you can choose to retweet it, comment on it, or do nothing. Make sure that you reshare or comment on at least one tweet from your feed. Reply to this by mentioning one of the actions you decided to take for every tweet. Follow this format: T1: do nothing, T2: reshare, T3: comment with 'WRITE YOUR COMMENT HERE'. Make sure you replace 'WRITE YOUR COMMENT HERE' with your comment, as a social media user, to the corresponding tweet."

def get_rating_prompt(attribute = "truth"):
    assert attribute in ["truth", "interest", "importance", "sentiment"], "Wrong rating attribute"
    if attribute == "interest":
        return "In the following, you will receive a list of statements. For every statement, your new task from now on will be to rate these statements according to how interesting you think they are. Your answer should be one of the following numbers 1: very uninteresting, 2: quite uninteresting, 3: slightly uninteresting, 4: neutral, 5: slightly interesting, 6: quite interesting, 7: very interesting. Only respond with the corresponding number. You don't need to explain your choice."
    elif attribute == "importance":
        return "In the following, you will receive a list of statements. For every statement, your new task from now on will be to rate these statements according to how important you think they are. Your answer should be one of the following numbers 1: very unimportant, 2: quite unimportant, 3: slightly unimportant, 4: neutral, 5: slightly important, 6: quite important, 7: very important. Only respond with the corresponding number. You don't need to explain your choice."
    elif attribute == "sentiment":
        return "In the following, you will receive a list of statements. For every statement, your new task from now on will be to rate these statements according to how cheerful you think they are. Your answer should be one of the following numbers 1: very sad, 2: quite sad, 3: slightly sad, 4: neutral, 5: slightly cheerful, 6: quite cheerful, 7: very cheerful. Only respond with the corresponding number. You don't need to explain your choice."
    return "In the following, you will receive a list of statements. For every statement, your new task from now on will be to rate these statements according to how true you perceive them to be. Your answer should be one of the following numbers 1: definitely false, 2: probably false, 3: possibly false, 4: uncertain, 5: possibly true, 6: probably true, 7: definitely true. Only respond with the corresponding number. You don't need to explain your choice."


def get_news_feed_exp_data():
    data = pd.read_csv("data/statements.csv")
    return data

def create_feed(target_statement, non_repeated_statements, start):
    if (start + feed_length - 1) > len(non_repeated_statements):
        claims_feed = pd.concat([non_repeated_statements.iloc[start:], non_repeated_statements.iloc[0: (start + feed_length - 1) % len(non_repeated_statements)]])
    else:
        claims_feed = non_repeated_statements.iloc[start : start + feed_length - 1]
    tweets = [pd.DataFrame({"statement_id": [claim.statement_id for claim in claims_feed.itertuples()],
                            "tweet_text": [claim.statement for claim in claims_feed.itertuples()],
                            "target": False,
                            "type": [claim.type for claim in claims_feed.itertuples()]})]
    tweets.insert(random.randint(0, feed_length - 1),
                  pd.DataFrame({"statement_id": [target_statement.statement_id],
                                "tweet_text": target_statement.statement,
                                "target": True,
                                "type": target_statement.type}))
    tweets = pd.concat(tweets)
    tweets["order_in_feed"] = [i for i in range(1, feed_length + 1)]
    return tweets

def init_all_news_feed(data, day_time, exp_folder = "experiments/"):
    # initialize experiment news_feeds
    # prepare all news feed for the experiment, every simulation has a corresponding target statement
    target_statements_subset = pd.concat([data.sample(nbr_simulations) for _ in range(10)]) # , random_state=2
    text_news_feed = []
    all_feed_content = []
    for nbr_sim, target_statement in enumerate(target_statements_subset.itertuples()):
        non_repeated_statements = data.loc[~data["statement_id"].isin([target_statement.statement_id])]
        # setting a seed for some reproducibility
        random.seed(nbr_sim)
        start_tweet_id = random.randint(0, len(non_repeated_statements))
        for nbr_iter in range(nbr_iterations):  # number of news feeds presented in one simulation
            feed_content = create_feed(target_statement, non_repeated_statements, start_tweet_id)

            start_tweet_id =  (start_tweet_id + (feed_length - 1) )% len(non_repeated_statements)  # to make sure we are not repeating non target claims in the feeds of one experiment
            prompt = ""
            for k, tweet in enumerate(feed_content.itertuples()):
                prompt += "T" + str(k + 1) + ": " + tweet.tweet_text + "\n"
                all_feed_content.append(pd.DataFrame(
                    {"simulation_num": nbr_sim,
                     "iteration_num": [nbr_iter],
                     "statement_id": tweet.statement_id,
                     "tweet": tweet.tweet_text,
                     "type": tweet.type,
                     "target": tweet.target}))
            text_news_feed.append(pd.DataFrame({"simulation_num": nbr_sim, "iteration_num": nbr_iter, "news_feed": [prompt]}))
    all_feed_content = pd.concat(all_feed_content).reset_index(drop=True)
    model_short_name = LLM_model.split("/")[-1] if "/" in LLM_model else LLM_model
    all_feed_content["model"] = model_short_name
    all_feed_content.to_csv(exp_folder + model_short_name + "_" + attribute + "_" + day_time + "_all_feed_content.csv", index=False)
    text_news_feed = pd.concat(text_news_feed).reset_index(drop=True)
    return text_news_feed, all_feed_content, target_statements_subset, nbr_simulations * 10

async def run_single_simulation_async(args):
    """Async version of run_single_simulation"""
    nbr_sim, target_statement, sim_feeds, claims_to_rate, system_prompt, trial_num = args

    client = AsyncOpenAI(api_key=openAI_key)

    local_feeds_actions = []
    rating = []
    log_buffer = []

    log_buffer.append(f"\n************ Simulation {nbr_sim} ************")

    # System prompt
    try:
        response = await client.responses.create(
            model=LLM_model,
            # reasoning={"effort": "minimal"},  # "minimal", "medium", or "high"
            input=system_prompt
        )
    except Exception as e:
        print(e)
        print(f"Error in system prompt: {e}")
        time.sleep(random.randint(30, 60))
        response = await client.responses.create(
            model=LLM_model,
            input=system_prompt
        )

    log_buffer.append(f"\n\nTARGET STATEMENT:{target_statement['statement']}")
    log_buffer.append(f"\n Type:{target_statement['type']}")

    for nbr_iter in range(nbr_iterations):
        log_buffer.append(f"\n-------------- Iteration {str(nbr_iter)} ---------------\n")
        feed_row = sim_feeds[sim_feeds["iteration_num"] == nbr_iter]
        if feed_row.empty:
            continue
        prompt = feed_row["news_feed"].iloc[0]

        log_buffer.append(f"\n\nFEED:\n{prompt}")
        try:
            response = await client.responses.create(
                model=LLM_model,
                input=prompt,
                previous_response_id=response.id
            )
        except Exception as e:
            print(f"Error in simulation feed loop: {e}")
            time.sleep(random.randint(30, 60))
            response = await client.responses.create(
                model=LLM_model,
                input=prompt,
                previous_response_id=response.id
            )

        log_buffer.append(f"\nASSISTANT:\n{response.output_text}")

        actions, comments = extract_actions(response.output_text, feed_length)

        local_feeds_actions.append(pd.DataFrame({
            "simulation_num": nbr_sim,
            "trial_num": trial_num,
            "iteration_num": nbr_iter,
            "action": actions,
            "comment": comments
        }))

    rating_prompt = get_rating_prompt(attribute)
    log_buffer.append(f"\n\nUSER:\n{rating_prompt}")
    try:
        response = await client.responses.create(
            model=LLM_model,
            input=rating_prompt,
            previous_response_id=response.id
        )
    except Exception as e:
        print(f"Error in rating prompt: {e}")
        time.sleep(random.randint(30, 60))
        response = await client.responses.create(
            model=LLM_model,
            input=rating_prompt,
            previous_response_id=response.id
        )

    log_buffer.append(f"\n\nASSISTANT: {response.output_text}")

    for statement in claims_to_rate.itertuples():
        log_buffer.append(f"\n\nUSER:\n{statement.tweet}")
        try:
            response = await client.responses.create(
                model=LLM_model,
                input=statement.tweet,
                previous_response_id=response.id
            )
        except Exception as e:
            print(f"Error in statement rating: {e}")
            time.sleep(random.randint(30, 60) )
            response = await client.responses.create(
                model=LLM_model,
                input=statement.tweet,
                previous_response_id=response.id
            )

        log_buffer.append(f"\n\nASSISTANT: {response.output_text}")
        rating.append(pd.DataFrame({
            "simulation_num": [nbr_sim],
            "trial_num": trial_num,
            "statement_id": statement.statement_id,
            "rating": response.output_text,
            "target": statement.target
        }))

    return local_feeds_actions, rating, "".join(log_buffer)


async def run_single_trial_async(trial_num, text_news_feed, all_feed_content, target_statement_subset, system_prompt):
    """Run a single trial asynchronously"""
    feeds_actions = []
    rating_statements = []
    trial_logs = []

    trial_logs.append(f"\n### Trial {trial_num} ###\n")

    tasks = []
    for nbr_sim, target_statement in enumerate(target_statement_subset.to_dict('records')):
        prompts_sim_feeds = text_news_feed[text_news_feed["simulation_num"] == nbr_sim]
        tweets_sim_feeds = all_feed_content[all_feed_content["simulation_num"] == nbr_sim]
        # get 3 unseen claims
        unseen_claims_to_rate = \
        all_feed_content[~all_feed_content["tweet"].isin(tweets_sim_feeds.tweet.unique())].sample(3, random_state=nbr_sim)[
            ["statement_id", "tweet"]]
        unseen_claims_to_rate["target"] = False
        repeated_claim = tweets_sim_feeds[tweets_sim_feeds["target"] == True][
            ["statement_id", "tweet", "target"]].drop_duplicates()
        # insert repeated claim in simulation
        claims_to_rate = pd.concat([unseen_claims_to_rate, repeated_claim]).sample(frac=1)
        tasks.append(run_single_simulation_async(
            (nbr_sim, target_statement, prompts_sim_feeds, claims_to_rate, system_prompt, trial_num)))

    # Run all simulations concurrently
    results = await asyncio_tqdm.gather(*tasks)

    for res_feeds, res_acc, res_log in results:
        if res_feeds:
            feeds_actions.extend(res_feeds)
        if res_acc:
            rating_statements.extend(res_acc)
        trial_logs.append(res_log)

    return trial_num, feeds_actions, rating_statements, "".join(trial_logs)


async def exp_news_feed_GPT(data, day_time):
    print("running experiment with repeated exposure...")
    
    model_name = LLM_model.split("/")[-1] if "/" in LLM_model else LLM_model
    exp_folder = f"experiments/{model_name}/{attribute}/"
    os.makedirs(exp_folder, exist_ok=True)

    all_feeds_actions = []
    all_rating_statements = []
    all_logs = []

    text_news_feed, all_feed_content, target_statement_subset, _ = init_all_news_feed(data, day_time, exp_folder)

    # Prepare tasks for each trial
    tasks = []
    for trial_num in range(nbr_trials):
        tasks.append(run_single_trial_async(
            trial_num, text_news_feed, all_feed_content, target_statement_subset, system_prompt
        ))

    # Run trials concurrently using asyncio
    print(f"Running {nbr_trials} trials asynchronously...")
    trial_results = await asyncio.gather(*tasks)

    # Collect results from all trials
    for trial_num, feeds_actions, rating_statements, trial_log in trial_results:
        print(f"Completed trial {trial_num}")
        all_feeds_actions.extend(feeds_actions)
        all_rating_statements.extend(rating_statements)
        all_logs.append(trial_log)

    # Save combined results
    model_filename = LLM_model.split("/")[-1] if "/" in LLM_model else LLM_model

    feeds_actions_df = pd.concat(all_feeds_actions) if all_feeds_actions else pd.DataFrame()
    feeds_actions_df["model"] = LLM_model
    feeds_actions_df.to_csv(
        exp_folder + model_filename + "_" + attribute + "_" + day_time + "_feeds_actions.csv", index=False)

    rating_statements_df = pd.concat(
        all_rating_statements) if all_rating_statements else pd.DataFrame()
    rating_statements_df["model"] = LLM_model
    rating_statements_df["attribute"] = attribute
    rating_statements_df.to_csv(
        exp_folder + model_filename + "_" + attribute + "_" + day_time + "_ratings.csv",
        index=False)

    # Write combined logs
    with open(exp_folder + model_filename + "_" + attribute + "_" + day_time + '_output.txt', 'w') as out_file:
        out_file.write("model:" + LLM_model + "\n")
        out_file.write(attribute + " rating")
        for log in all_logs:
            out_file.write(log)

    return "END experiment news feed GPT"

def transformers_models_response(pipe, tokenizer, messages_list):
    """
    Processes a batch of conversations with a Hugging Face pipeline.

    Args:
        pipe: The text-generation pipeline.
        tokenizer: The tokenizer.
        messages_list: A list of message lists, where each inner list is a conversation.
        **kwargs: Additional arguments passed to the pipeline (e.g., generation parameters).

    Returns:
        A list of generated responses.
    """
    outputs = pipe(messages_list, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id, max_length=None, temperature=temperature, do_sample=True)
    return [output[0]["generated_text"][-1]["content"] for output in outputs]


def exp_news_feed_open_weight(data, day_time):
    print("running experiment with repeated exposure...")

    model_name = LLM_model.split("/")[-1] if "/" in LLM_model else LLM_model
    exp_folder = f"experiments/{model_name}/temp_{temperature}/{attribute}/"
    os.makedirs(exp_folder, exist_ok=True)

    text_news_feed, all_feed_content, target_statements_subset, nbr_sims = init_all_news_feed(data, day_time, exp_folder)

    # Check available GPUs
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(
            f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB")


    # More explicit device mapping for FP8 models
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        torch.mps.empty_cache()
    else:
        device = "cpu"
        print("Warning: Loading FP8 model on CPU. Performance will be suboptimal.")
    print("Device:", device)
    tokenizer = AutoTokenizer.from_pretrained(LLM_model,
                                              cache_dir=CUSTOM_MODEL_CACHE_DIR)
    # Set pad_token to eos_token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    pipe = pipeline(
        "text-generation",
        model=LLM_model,
        tokenizer=tokenizer,
        model_kwargs={
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "attn_implementation": "flash_attention_2",
            "max_memory": {
                0: "130GB",  # Leave some headroom on each H200
                1: "130GB",
                "cpu": "50GB"
            },
            "cache_dir": CUSTOM_MODEL_CACHE_DIR,
        },
        device=device
    )

    # Fix for warning: "Both `max_new_tokens` and `max_length` seem to have been set."
    # We unset max_length in the generation config to avoid the conflict and the print overhead.
    pipe.model.generation_config.max_length = None
    # Also unset in the main config to prevent fallback values (often 20) from triggering the warning
    if hasattr(pipe.model.config, "max_length"):
        pipe.model.config.max_length = None
    # Suppress only the generation utils logger warning (keeps other transformers warnings visible)
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

    base_news_feed = [[{"role": "system", "content": system_prompt}]]
    repeated_claims = {}
    news_feed = []
    for _ in range(nbr_sims * nbr_trials):
        news_feed.extend(copy.deepcopy(base_news_feed))

    for nbr_iter in tqdm.tqdm(range(nbr_iterations)):
        # batch append conversation with new feed
        for i in range(nbr_sims * nbr_trials):
            prompt_sim_feed = text_news_feed.loc[(text_news_feed["simulation_num"] == (i % nbr_sims)) & (
                        text_news_feed["iteration_num"] == nbr_iter), "news_feed"].iloc[0]
            news_feed[i].append({"role": "user", "content": prompt_sim_feed})
        # get response
        # responses = ["test" for _ in range(nbr_sims * nbr_trials)]
        responses = transformers_models_response(pipe=pipe, tokenizer=tokenizer, messages_list=news_feed)
        # append batch responses
        for i in range(nbr_sims * nbr_trials):
            news_feed[i].append({"role": "assistant", "content": responses[i]})

    # add statements for rating
    for i in range(nbr_sims * nbr_trials):
        # index = i % nbr_sims
        news_feed[i].append({"role": "user", "content": get_rating_prompt(attribute)})
    # responses = ["test" for _ in range(nbr_sims * nbr_trials)]
    responses = transformers_models_response(pipe=pipe, tokenizer=tokenizer, messages_list=news_feed)
    # append batch responses
    for i in range(nbr_sims * nbr_trials):
        news_feed[i].append({"role": "assistant", "content": responses[i]})

    for unseen_claim_it in tqdm.tqdm(range(3 + 1)):
        for i in range(nbr_sims * nbr_trials):
            tweets_sim_feeds = all_feed_content[all_feed_content["simulation_num"] == i % nbr_sims][["statement_id", "tweet", "target", "type", "simulation_num"]].drop_duplicates()#.sample(frac = 1, random_state = i % nbr_sim_scenario)
            repeated_claim_feeds = tweets_sim_feeds[tweets_sim_feeds["target"] == True]
            repeated_claims[i % nbr_sims] = repeated_claim_feeds
            # get claims not in feeds
            random.seed(i % nbr_sims)
            claims_to_rate = random.sample(list(all_feed_content[~all_feed_content["tweet"].isin(tweets_sim_feeds.tweet.unique())]["tweet"].unique()), 3)
            # insert the repeated claim
            claims_to_rate.insert(random.randint(0, 3), repeated_claim_feeds.tweet.iloc[0])
            news_feed[i].append({"role": "user", "content": claims_to_rate[unseen_claim_it]})
        # get response
        # responses = [random.randint(0, 4) for _ in range(nbr_sims * nbr_trials)]
        responses = transformers_models_response(pipe=pipe, tokenizer=tokenizer, messages_list=news_feed)
        # append batch responses
        for i in range(nbr_sims * nbr_trials):
            news_feed[i].append({"role": "assistant", "content": responses[i]})

    statements = pd.concat(
        [pd.DataFrame(iteration_news_feed for iteration_news_feed in news_feed_instance) for news_feed_instance in
         news_feed])
    # statements.to_csv("debug_statement.csv", index=False)

    # statements = pd.read_csv("debug_statement.csv")
    nbr_sim_list = []
    # nbr_iteration * (user, assistant) + (system, accuracy_prompt, assistant_response) + (3 unseen claims to rate + 1 repeated claim to rate) * (user, assistant)
    for elem in [[i] * (nbr_iterations * 2 + 3 + (3 + 1) * 2) for i in range(nbr_sims)] * nbr_trials:
        nbr_sim_list += elem
    statements["simulation_num"] = nbr_sim_list
    num_trial_list = []
    for i in range(nbr_trials):
        for j in range(nbr_sims):
            for k in range(nbr_iterations * 2 + 3 + (3 + 1) * 2):
                num_trial_list.append(i)
    statements["trial_num"] = num_trial_list

    num_iter_list = ["system_prompt"]
    for i in range(nbr_iterations):
        num_iter_list += [i, i]
    num_iter_list += ["rating_task", "assistant_comment"]
    for _ in range(3 + 1):
        num_iter_list += ["claim", "rating"]
    num_iter_list = num_iter_list * nbr_sims * nbr_trials
    statements["iteration_num"] = num_iter_list
    repeated_claims = pd.concat(repeated_claims).reset_index(drop = True)
    statements["model"] = LLM_model
    statements["attribute"] = attribute
    statements = statements.merge(data, left_on=["content"], right_on=["statement"], how="left")
    statements =  statements.merge(repeated_claims[["tweet", "simulation_num", "target"]], left_on=["content", "simulation_num"], right_on=["tweet", "simulation_num"], how="left")

    # save conversation results
    model_short_name = LLM_model.split("/")[-1] if "/" in LLM_model else LLM_model
    statements.to_csv(exp_folder + model_short_name + "_" + attribute + "_" + day_time + "_simulation_results.csv", index=False)

    # conversation log
    with open(exp_folder + model_short_name + "_" + attribute + "_" + day_time + '_output.txt', 'w') as out_file:
        out_file.write("model: " + LLM_model + "\n")
        out_file.write(attribute + " rating")
        for nbr_sim, sim_res in enumerate(news_feed):
            out_file.write("\n\n********************************\nSimulation: " + str(
                nbr_sim % nbr_sims) + " --- Trial: " + str(
                nbr_sim % nbr_trials) + " ---\n********************************\n\n")
            out_file.write(
                f"\nTARGET STATEMENT: {all_feed_content[(all_feed_content['simulation_num'] == (nbr_sim % nbr_sims)) & (all_feed_content['target'] == True)].tweet.iloc[0]}")
            out_file.write(
                f"\nTYPE: {all_feed_content[(all_feed_content['simulation_num'] == (nbr_sim % nbr_sims)) & (all_feed_content['target'] == True)].type.iloc[0]}\n\n")
            for it, res in enumerate(sim_res):
                if it % 2 == 1:
                    out_file.write("\n\n---------- Iteration " + str((it // 2)) + " ----------\n\n")
                out_file.write(res["role"].upper() + ": " + str(res["content"]) + "\n")
    return "DONE exp_news_feed_open_weight"

# uncomment Comment these lines if you want to use Slack notification when the code finish running on cluster
# webhook_url = "ADD YOUR WEBHOOK URL"
# @slack_sender(webhook_url=webhook_url, channel="code_notification")
def run_experiment():
    '''
    :param LLM_model: the name of the model we want to use in the experiment
    :param nbr_simulations: the number of target claims we test in the experiment
    :param nbr_iterations: the number of news feed with the target repeated claim we are showing one instance of LLM
    :param feed_length: the number of tweet in a single news feed given to the model
    :return:
    '''
    day_time = get_daytime()
    print("output_files timestamp:", day_time)
    data = get_news_feed_exp_data()

    if "gpt" in LLM_model:
        asyncio.run(exp_news_feed_GPT(data ,day_time))
    else:
        # Synchronous for HF models
        print("temperature:", temperature)
        exp_news_feed_open_weight(data, day_time)

    return "DONE experiment news feed " + LLM_model + ", attribute " + attribute + ", temperature " + str(temperature) + ", nbr_simulations " + str(nbr_simulations * 10)


if __name__ == "__main__":
    print("model:", LLM_model)
    print("nbr_simulations:", nbr_simulations)
    print("nbr_trials:", nbr_trials)
    print("attribute:", attribute)
    run_experiment()
