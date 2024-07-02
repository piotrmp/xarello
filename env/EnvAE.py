import numpy as np
import time
from gymnasium import Env
from gymnasium import spaces
from tokenizers.decoders import WordPiece
from transformers import AutoTokenizer, AutoConfig

from utilities.llm_utils import get_best_candidates, get_candidate_embeddings_llm, get_candidate_embeddings_static


class EnvAE(Env):
    spec = None
    metadata = {}
    reward_range = (-1.0, 1.0)
    MAX_TEXT_LENGTH = 512
    STEPS_PER_EPISODE = 5
    EPISODES_PER_TEXT = 5
    BEST_K = 20
    
    def __init__(self, pretrained_model, texts, victim, device, static_embedding=False, protected_tokens=[]):
        # Global settings
        self.tokeniser = AutoTokenizer.from_pretrained(pretrained_model)
        self.vocab_size = AutoConfig.from_pretrained(pretrained_model).vocab_size
        self.action_space = spaces.Tuple((spaces.Discrete(self.MAX_TEXT_LENGTH), spaces.Discrete(self.vocab_size)))
        
        self.victim = victim
        self.orig_texts = texts
        tokenised = self.tokeniser(texts, padding='max_length', truncation=True, max_length=self.MAX_TEXT_LENGTH,
                                   return_offsets_mapping=True)
        self.orig_tokens = np.array(tokenised['input_ids'])
        self.overflow = [(texts[i][x[-2][1]:] if x[-2][1] != 0 else '') for i, x in
                         enumerate(tokenised['offset_mapping'])]
        if np.sum([len(o) > 0 for o in self.overflow]) > 0:
            print("Warning, " + str(
                np.sum([len(o) > 0 for o in self.overflow])) + " fragments longer than max text length.")
        start = time.time()
        self.replacement_tokens = get_best_candidates(self.orig_texts, self.orig_tokens, self.BEST_K, self.tokeniser,
                                                      pretrained_model, device, protected_tokens=protected_tokens)
        duration = time.time() - start
        print("Prepared candidates in " + str(duration) + " s.")
        start = time.time()
        if not static_embedding:
            self.replacement_embeddings = get_candidate_embeddings_llm(self.orig_texts, self.orig_tokens,
                                                                       self.replacement_tokens, self.BEST_K,
                                                                       self.tokeniser,
                                                                       pretrained_model, device)
            self.candidate_embedding_size = AutoConfig.from_pretrained(pretrained_model).hidden_size
        else:
            self.replacement_embeddings = get_candidate_embeddings_static(self.replacement_tokens, self.tokeniser)
            self.candidate_embedding_size = 300
        duration = time.time() - start
        print("Prepared embeddings in " + str(duration) + " s.")
        self.decoder = WordPiece()
        self.current_text = -1
        self.current_tokens = None
        self.current_steps = -1
        self.previous_steps = -1
        self.episode = -1
        self.proceed_next = True
        # self.observation_space = spaces.MultiDiscrete(
        #    np.array([self.vocab_size] * self.MAX_TEXT_LENGTH))
        self.observation_space = spaces.Tuple((spaces.Discrete(len(texts)), spaces.Discrete(2),
                                               spaces.MultiDiscrete(
                                                   np.array([self.vocab_size] * self.MAX_TEXT_LENGTH))))
    
    def step(self, action_tuple):
        action_location, action_replacement = action_tuple
        # Take the action and modify the situation
        self.current_steps += 1
        truncated = (self.current_steps % self.STEPS_PER_EPISODE == 0)
        if self.current_tokens[action_location] in [self.tokeniser.pad_token_id, self.tokeniser.sep_token_id,
                                                    self.tokeniser.cls_token_id]:
            # adding tokens in padding area is not allowed, -1 penalty
            # also reset the sentence
            truncated = True
            print("ACTION " + str(action_tuple) + ": " +
                  self.tokeniser.convert_ids_to_tokens([self.current_tokens[action_location]])[
                      0] + " -> " + self.tokeniser.convert_ids_to_tokens([self.current_tokens[action_location]])[
                      0] + " reward: " + str(-1))
            return (self.current_text, self.orig_prediction, self.current_tokens), -1.0, False, truncated, {}
        previous_tokens = self.current_tokens.copy()
        replacement_token = self.replacement_tokens[self.current_text][action_location][action_replacement]
        self.current_tokens[action_location] = replacement_token
        prediction_change = self.victim.pred_changed(self.orig_tokens[self.current_text], self.current_tokens,
                                                     previous_tokens)
        if isinstance(prediction_change, bool):
            reward = 1.0 if prediction_change else 0.0
            terminated = prediction_change
        else:
            reward = prediction_change
            terminated = (prediction_change == 1.0)
        # if terminated:
        #    self.proceed_next = True
        print("ACTION " + str(action_tuple) + ": " +
              self.tokeniser.convert_ids_to_tokens([previous_tokens[action_location]])[0] + " -> " +
              self.tokeniser.convert_ids_to_tokens([replacement_token])[0] + " reward: " + str(
            reward))
        return (self.current_text, self.orig_prediction, self.current_tokens), reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode = self.episode + 1
        if self.episode == self.EPISODES_PER_TEXT or self.proceed_next:
            self.proceed_next = False
            self.current_text = (self.current_text + 1) % len(self.orig_texts)
            self.previous_steps = self.current_steps
            self.current_steps = 0
            self.orig_prediction = self.victim.pred(self.orig_tokens[self.current_text])
            self.episode = 0
        self.current_tokens = self.orig_tokens[self.current_text].copy()
        # if self.victim.pred(self.orig_tokens[self.current_i]) == 0.0:
        #    return self.reset(seed, options)
        # else:
        return ((self.current_text, self.orig_prediction, self.current_tokens), {})
    
    def render(self):
        # Render visualisation
        result = 'CURRENT TEXT: ' + self.decoder.decode(
            self.tokeniser.convert_ids_to_tokens(self.current_tokens)).replace(' [PAD]', '')
        return result
    
    def close(self):
        # shut down
        pass


def copy_observation(observation):
    return (observation[0], observation[1], observation[2].copy())
