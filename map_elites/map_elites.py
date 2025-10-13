import subprocess
import numpy as np
import json
import tiktoken
from datetime import datetime
import os
from utils.utils import *
from map_elites.gls_tsp_adapt.gls_tsp_eval import Sandbox
from embedding_cluster import *
import time

class Map_Elites:
    def __init__(self, cfg, root_dir) -> None:
        self.cfg = cfg
        self.root_dir = root_dir

        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.generation = 0
        self.function_evals = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_request = 0
        self.best_obj_overall = float("inf")
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None

        self.init_prompt()
        self.init_population()

    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type

        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)
        logging.info("Function name: " + self.func_name)
        logging.info("Stop condition: " + self.cfg.stop_condition)

        self.prompt_dir = f"{self.root_dir}/map_elites/prompts"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"

        # Loading all text prompts
        # Problem-specific prompt components
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'
        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt')
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        if os.path.exists(f'{problem_prompt_path}/external_knowledge.txt'):
            self.external_knowledge = file_to_string(f'{problem_prompt_path}/external_knowledge.txt')
            self.long_term_reflection_str = self.external_knowledge
        else:
            self.external_knowledge = ""

        # Common prompts
        self.system_generator_prompt = file_to_string(f'{self.prompt_dir}/common/system_generator.txt')
        self.system_reflector_prompt = file_to_string(f'{self.prompt_dir}/common/system_reflector.txt')
        self.user_reflector_st_prompt = file_to_string(
            f'{self.prompt_dir}/common/user_reflector_st.txt') if self.problem_type != "black_box" else file_to_string(
            f'{self.prompt_dir}/common/user_reflector_st_black_box.txt')  # short-term reflection
        self.user_reflector_lt_prompt = file_to_string(
            f'{self.prompt_dir}/common/user_reflector_lt.txt')  # long-term reflection
        self.crossover_prompt = file_to_string(f'{self.prompt_dir}/common/crossover.txt')
        self.mutataion_prompt = file_to_string(f'{self.prompt_dir}/common/mutation.txt')
        self.user_generator_prompt = file_to_string(f'{self.prompt_dir}/common/user_generator.txt').format(
            func_name=self.func_name,
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
        )
        self.init_user_generator_prompt = file_to_string(f'{self.prompt_dir}/common/init_user_generator.txt')
        self.seed_prompt = file_to_string(f'{self.prompt_dir}/common/seed.txt').format(
            seed_func=self.seed_func,
            func_name=self.func_name,
        )

        # Flag to print prompts
        self.print_crossover_prompt = True  # Print crossover prompt for the first iteration
        self.print_mutate_prompt = True  # Print mutate prompt for the first iteration
        self.print_short_term_reflection_prompt = True  # Print short-term reflection prompt for the first iteration
        self.print_long_term_reflection_prompt = True  # Print long-term reflection prompt for the first iteration

        self.strategies = self.cfg.problem.strategies

        _cur_file_ = os.path.dirname(__file__)
        _cur_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # self._my_log_path = os.path.join(_cur_file_, 'all_logs', f'{self.cfg.problem.problem_name}_{_cur_timestamp}')
        # os.makedirs(self._my_log_path, exist_ok=True)

    def cal_usage_LLM(self, lst_prompt, lst_completion, encoding_name="cl100k_base"):
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        self.llm_request += len(lst_prompt)
        for i in range(len(lst_prompt)):
            for message in lst_prompt[i]:
                for key, value in message.items():
                    self.prompt_tokens += len(encoding.encode(value))

            self.completion_tokens += len(encoding.encode(lst_completion[i]))

    def init_population(self) -> None:
        # Evaluate the seed function, and set it as Elite
        logging.info("Evaluating seed function...")
        code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
        logging.info("Seed function code: \n" + code)
        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
            "code_path": f"problem_iter{self.iteration}_code0.py",
            "code": code,
            "response_id": 0,
        }
        self.seed_ind = seed_ind
        self.population = self.evaluate_population([seed_ind])

        # If the seed function is invalid, stop
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")

        self.update_iter()

        # Generate responses
        messages_lst = []
        for i in range(self.cfg.init_pop_size):
            user_generator_prompt_full = self.init_user_generator_prompt.format(
                strategy=self.strategies[i % len(self.strategies)],
                func_name=self.func_name,
                problem_desc=self.problem_desc,
                func_desc=self.func_desc,
            )

            system_generator_prompt_full = self.system_generator_prompt

            # Generate responses
            system = system_generator_prompt_full
            user = user_generator_prompt_full + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str

            pre_messages = {"system": system, "user": user}
            messages = format_messages(self.cfg, pre_messages)
            messages_lst.append(messages)

            logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)

            # Write to file
            file_name = f"problem_iter{self.iteration}_prompt{i}.txt"
            with open(file_name, 'w') as file:
                file.writelines(json.dumps(pre_messages))

        responses = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature + 0.3)
        self.cal_usage_LLM(messages_lst, responses)
        '''responses = multi_chat_completion([messages], self.cfg.init_pop_size, self.cfg.model,
                                          self.cfg.temperature + 0.3)  # Increase the temperature for diverse initial population'''
        population = [self.response_to_individual(response, response_id) for response_id, response in
                      enumerate(responses)]

        # Run code and evaluate population
        population = self.evaluate_population(population)
        self.archive_evaluated_population(population)

        # Update iteration
        self.update_iter()
        self.update_generation()

    def response_to_individual(self, response: str, response_id: int, file_name: str = None) -> dict:
        """
        Convert response to individual
        """
        # Write response to file
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w') as file:
            file.writelines(response + '\n')

        code = extract_code_from_generator(response)

        # Extract code and description from response
        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"

        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "code": code,
            "response_id": response_id,
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["gap"] = float("inf")
        individual["runtime"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual

    def evaluate_population(self, population: list[dict], hs_try_idx: int = None) -> list[dict]: # type: ignore
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        inner_runs = []

        # Run code to evaluate
        for response_id in range(len(population)):
            self.function_evals += 1
            # Skip if response is invalid
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
                inner_runs.append(None)
                continue

            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")

            if self.problem == 'tsp_gls':
                try:
                    # Use sandboxed execution for 'tsp_gls'
                    sandbox = Sandbox()
                    result, run_ok = sandbox.run(population[response_id]['code'])
                    inner_runs.append((result, run_ok))
                except Exception as e:  # If sandbox execution fails
                    logging.info(f"Error for response_id {response_id}: {e}")
                    population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                    inner_runs.append(None)
            else:
                try:
                    # Use default code execution for other problems
                    process = self._run_code(population[response_id], response_id)
                    inner_runs.append(process)
                except Exception as e:  # If code execution fails
                    logging.info(f"Error for response_id {response_id}: {e}")
                    population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                    inner_runs.append(None)

        # Update population with objective values
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None:  # If code execution fails, skip
                continue

            individual = population[response_id]

            if self.problem == 'tsp_gls':
                result, run_ok = inner_run
                if run_ok:
                    try:
                        individual["gap"] = float(result) if self.obj_type == "min" else -float(result)
                        individual["exec_success"] = True
                    except:
                        population[response_id] = self.mark_invalid_individual(population[response_id],
                                                                               "Invalid objective value!")
                else:
                    population[response_id] = self.mark_invalid_individual(population[response_id],
                                                                           "Sandbox execution failed!")
            else:    
                try:
                    inner_run.communicate(timeout=self.cfg.timeout)  # Wait for code execution to finish
                except subprocess.TimeoutExpired as e:
                    logging.info(f"Error for response_id {response_id}: {e}")
                    population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                    inner_run.kill()
                    continue

                stdout_filepath = individual["stdout_filepath"]
                with open(stdout_filepath, 'r') as f:  # read the stdout file
                    stdout_str = f.read()
                traceback_msg = filter_traceback(stdout_str)

                if traceback_msg == '':  # If execution has no error
                    try:
                        # Split the output into lines
                        lines = stdout_str.strip().split('\n')
                        print("debug 1")
                        individual["gap"] = float(lines[-2]) if self.obj_type == "min" else -float(lines[-2])
                        print("debug 2")

                        # Extract runtime from the second-to-last line
                        runtime_line = lines[-1]

                        parts = runtime_line.split()
                        print("Parts:", parts)

                        # user_time = float(parts[0].replace("user", ""))
                        # system_time = float(parts[1].replace("system", ""))
                        print("debug 3")
                        for i in range(len(parts)):
                            print(f"part{i}: {parts[i]}") 

                        individual["runtime"] = float(parts[2]) + float(parts[4])
                        
                        print("debug 4")

                        self.get_embedding(individual)
                        
                        print("debug 5")
                        
                        individual["exec_success"] = True
                    except:
                        print("debug here guys")
                        population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid std out / objective value!")
                else:  # Otherwise, also provide execution traceback error feedback
                    population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['gap']}, run_time: {individual['runtime']}")
        return population
    
    def _run_code(self, individual: dict, response_id) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")

        with open(self.output_file, 'w') as file:
            file.writelines(individual["code"] + '\n')

        # Execute the python file with flags
        with open(individual["stdout_filepath"], 'w') as f:
            eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py' if self.problem_type != "black_box" else f'{self.root_dir}/problems/{self.problem}/eval_black_box.py'
            process = subprocess.Popen(['time','python', '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train"],
                                       stdout=f, stderr=f)

        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration,
                            response_id=response_id)
        
        return process
    
    
    def get_embedding(self, individual: dict):
        print("debug 6")
        code = individual["code"]
        print("debug 7")
        code_clean = remove_comments_and_docstrings(code)
        print("debug 8")
        code_pep8 = standardize_code(code_clean)
        print("debug 9")
        individual["embedding"] = get_nvidia_embedding(code_pep8, self.cfg.embedding_model)
        print("debug 10")

    def is_dominated(self, ind_a, ind_b, obj_type="min"):
        """
        Returns True if ind_a is dominated by ind_b considering both 'gap' and 'runtime'.
        """
        a_values = [ind_a["gap"], ind_a["runtime"]]
        b_values = [ind_b["gap"], ind_b["runtime"]]
        return all(b <= a for a, b in zip(a_values, b_values)) and any(b < a for a, b in zip(a_values, b_values))
    
    def archive_evaluated_population(self, evaluated_population: list[dict]) -> None:
        # Only consider successful individuals
        valid_individuals = [ind for ind in evaluated_population if ind.get("exec_success", False)]
        # Combine current population and new valid individuals
        combined_population = self.population + valid_individuals
        combined_embeddings = [ind["embedding"] for ind in combined_population]
        clusters = []  # Each cluster is a list of indices in combined_population

        for idx, emb in enumerate(combined_embeddings):
            assigned = False
            for cluster in clusters:
                # Check similarity with all members in the cluster
                sims = [cosine_similarity([emb], [combined_embeddings[member_idx]])[0][0] for member_idx in cluster]
                if all(sim > self.cfg.alpha for sim in sims):
                    cluster.append(idx)
                    assigned = True
                    break
            if not assigned:
                clusters.append([idx])

        # Keep only non-dominated individuals in each cluster
        non_dominated_individuals = []

        for cluster in clusters:
            cluster_inds = [combined_population[i] for i in cluster]
            nd_inds = []

            for ind in cluster_inds:
                dominated = False
                to_remove = []
                for nd in nd_inds:
                    if self.is_dominated(ind, nd, self.obj_type):
                        dominated = True
                        break
                    elif self.is_dominated(nd, ind, self.obj_type):
                        to_remove.append(nd)
                # remove any existing dominated individual
                for r in to_remove:
                    nd_inds.remove(r)
                if not dominated:
                    nd_inds.append(ind)

            non_dominated_individuals.extend(nd_inds)

        # Update population with best individuals from clusters
        self.population = non_dominated_individuals

    def get_non_dominated_individuals(self):
        """
        Returns a list of non-dominated individuals from self.population.
        """
        non_dominated = []
        for i, ind_a in enumerate(self.population):
            dominated = False
            for j, ind_b in enumerate(self.population):
                if i != j and self.is_dominated(ind_a, ind_b, self.obj_type):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(ind_a)
        return non_dominated

    def update_iter(self) -> None:
        """
        Update after each iteration
        """
        population = self.population
        objs = [individual["gap"] for individual in population]
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))

        # update best overall
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]

        # Dump the current population to a JSON file for inspection
        population_to_dump = [
            {k: v for k, v in individual.items() if k != "embedding"}
            for individual in self.population
        ]
        with open(f"population_iter{self.iteration}.json", "w") as f:
            json.dump(population_to_dump, f, indent=2)

        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"LLM usage: prompt_tokens = {self.prompt_tokens}, completion_tokens = {self.completion_tokens}")
        logging.info(f"LLM Requests: {self.llm_request}")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1

    def update_generation(self) -> None:
        logging.info(f"Generation {self.generation} finished...")
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"LLM usage: prompt_tokens = {self.prompt_tokens}, completion_tokens = {self.completion_tokens}")
        logging.info(f"LLM Requests: {self.llm_request}")
        logging.info(f"Function Evals: {self.function_evals}")
        self.generation += 1

    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        if self.problem_type == "black_box":
            population = [individual for individual in population if
                          individual["exec_success"] and individual["gap"] < self.seed_ind["gap"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            if parents[0]["gap"] != parents[1]["gap"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def gen_short_term_reflection_prompt(self, ind1: dict, ind2: dict) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """
        if ind1["gap"] == ind2["gap"]:
            print(ind1["code"], ind2["code"])
            raise ValueError("Two individuals to crossover have the same objective value!")
        # Determine which individual is better or worse
        if ind1["gap"] < ind2["gap"]:
            better_ind, worse_ind = ind1, ind2
        elif ind1["gap"] > ind2["gap"]:
            better_ind, worse_ind = ind2, ind1

        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])

        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name=self.func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            worse_code=worse_code,
            better_code=better_code
        )

        pre_messages = {"system": system, "user": user}
        message = format_messages(self.cfg, pre_messages)

        # Print reflection prompt for the first iteration
        if self.print_short_term_reflection_prompt:
            logging.info("Short-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_short_term_reflection_prompt = False
        return message, worse_code, better_code

    def short_term_reflection(self, population: list[dict]) -> tuple[list[list[dict]], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i + 1]

            # Short-term reflection
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)

        # Asynchronously generate responses
        response_lst = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature)
        self.cal_usage_LLM(messages_lst, response_lst)
        return response_lst, worse_code_lst, better_code_lst

    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """
        Long-term reflection before mutation.
        """
        system = self.system_reflector_prompt
        user = self.user_reflector_lt_prompt.format(
            problem_desc=self.problem_desc,
            prior_reflection=self.long_term_reflection_str,
            new_reflection="\n".join(short_term_reflections),
        )

        pre_messages = {"system": system, "user": user}
        messages = format_messages(self.cfg, pre_messages)

        if self.print_long_term_reflection_prompt:
            logging.info("Long-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_long_term_reflection_prompt = False

        self.long_term_reflection_str = multi_chat_completion([messages], 1, self.cfg.model, self.cfg.temperature)[0]
        self.cal_usage_LLM([messages], [self.long_term_reflection_str])
        # Write reflections to file
        file_name = f"problem_iter{self.iteration}_short_term_reflections.txt"
        with open(file_name, 'w') as file:
            file.writelines("\n".join(short_term_reflections) + '\n')

        file_name = f"problem_iter{self.iteration}_long_term_reflection.txt"
        with open(file_name, 'w') as file:
            file.writelines(self.long_term_reflection_str + '\n')

    def crossover(self, short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]) -> list[dict]:
        reflection_content_lst, worse_code_lst, better_code_lst = short_term_reflection_tuple
        messages_lst = []
        num_choice = 0
        for reflection, worse_code, better_code in zip(reflection_content_lst, worse_code_lst, better_code_lst):
            # Crossover
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.crossover_prompt.format(
                user_generator=self.user_generator_prompt,
                func_signature0=func_signature0,
                func_signature1=func_signature1,
                worse_code=worse_code,
                better_code=better_code,
                reflection=reflection,
                func_name=self.func_name,
            )

            pre_messages = {"system": system, "user": user}
            messages = format_messages(self.cfg, pre_messages)
            messages_lst.append(messages)

            # Write to file
            file_name = f"problem_iter{self.iteration}_response{num_choice}_prompt.txt"
            with open(file_name, 'w') as file:
                file.writelines(json.dumps(pre_messages))
            num_choice += 1

            # Print crossover prompt for the first iteration
            if self.print_crossover_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_crossover_prompt = False

        # Asynchronously generate responses
        response_lst = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature)
        self.cal_usage_LLM(messages_lst, response_lst)
        crossed_population = [self.response_to_individual(response, response_id) for response_id, response in
                              enumerate(response_lst)]

        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population

    def mutate(self, population: list[dict]) -> list[dict]:
        system = self.system_generator_prompt
        func_signature1 = self.func_signature.format(version=1)
        n = len(population)
        messages_lst = []
        for i in range(n):
            user = self.mutataion_prompt.format(
                func_signature1=func_signature1,
                code=filter_code(population[i]["code"]),
                func_name=self.func_name,
            )

            pre_messages = {"system": system, "user": user}
            messages = format_messages(self.cfg, pre_messages)
            messages_lst.append(messages)

        # Write to file
        file_name = f"problem_iter{self.iteration}_prompt.txt"
        with open(file_name, 'w') as file:
            file.writelines(json.dumps(pre_messages))

        if self.print_mutate_prompt:
            logging.info("Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_mutate_prompt = False
        responses = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature)
        self.cal_usage_LLM(messages_lst, responses)
        population = [self.response_to_individual(response, response_id) for response_id, response in
                      enumerate(responses)]
        return population

    def stop(self) -> bool:
        """
        Check if the stopping condition is met, based on config.stop_condition.
        Supported conditions: 'token', 'fe', 'gen'
        """
        cond = self.cfg.stop_condition
        if cond == "token":
            return (self.prompt_tokens + self.completion_tokens) >= self.cfg.max_token
        elif cond == "fe":
            return self.function_evals >= self.cfg.max_fe
        elif cond == "gen":
            return self.generation >= self.cfg.max_gen
        else:
            # Default: stop if max_token reached
            return (self.prompt_tokens + self.completion_tokens) >= self.cfg.max_token
        
    def evolve(self):
        while not self.stop():
            # If all individuals are invalid, stop
            if all([not individual["exec_success"] for individual in self.population]):
                raise RuntimeError(f"All individuals are invalid. Please check the stdout files in {os.getcwd()}.")
            # Select
            population_to_select = self.population 
            selected_population = self.random_select(population_to_select)
            if selected_population is None:
                raise RuntimeError("Selection failed. Please check the population.")
            # Short-term reflection
            short_term_reflection_tuple = self.short_term_reflection(selected_population)  # (response_lst, worse_code_lst, better_code_lst)
            # Crossover
            crossed_population = self.crossover(short_term_reflection_tuple)
            # Evaluate
            evaluated_population = self.evaluate_population(crossed_population)
            # Update
            self.update_iter()
            # Mutate
            mutated_population = self.mutate(evaluated_population)
            # Evaluate
            evaluated_population = self.evaluate_population(mutated_population)
            self.archive_evaluated_population(evaluated_population)
            # Update
            self.update_iter()
            self.update_generation()

        logging.info(f"Token used: {(self.prompt_tokens + self.completion_tokens)}.")
        return self.best_code_overall, self.best_code_path_overall