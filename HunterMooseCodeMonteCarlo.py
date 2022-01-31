#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: evgeniy
"""
from HunterMooseCode import PlayerBasic

from collections import Counter
import random


class PlayerMC(PlayerBasic):

    def __init__(self, label, 
                 current_pos, 
                 moves,
                 start_limit=None,
                 end_limit=None):
        super(PlayerMC, self).__init__(label, 
                                     current_pos, 
                                     moves, 
                                     start_limit, 
                                     end_limit)

    def flush(self):
        self.current_pos = self.start_pos

       
class MCGameSimulator(object):
    
    def __init__(self, hunter, prey, num_simulation_games=10000):
        assert len(set(hunter.moves) & set(prey.moves)) == 0, \
               "The die choices for hunter and prey should not meet."
        self.hunter = hunter
        self.prey = prey
        self.num_games = num_simulation_games
    
        self.hunter_nmb_wins = 0
        self.hunter_stats_win = Counter()

        self.prey_nmb_wins = 0
        self.prey_stats_win = Counter()
        
        self.cnt_rolls = 0
        self.stats_end_game = Counter()
       
    @property
    def steps(self):
        return len(self.stats_end_game)
        
    def print_stats(self):
        print("------------------------------------")
        for step in range(1, self.steps+1):
            print("Step %d:    Hunter: %.5f, Prey: %.5f, End game: %.5f" % \
                  (step, self.hunter_stats_win[step], \
                         self.prey_stats_win[step], \
                         self.stats_end_game[step]))
        print("--------")
        
        print("Overall:    Hunter: %.5f, Prey: %.5f, End game: %.5f" % \
                       ( sum(self.hunter_stats_win.values()), \
                         sum(self.prey_stats_win.values()), \
                         sum(self.stats_end_game.values()) )   )
        print("------------------------------------")
        
    def flush_players(self):
        self.hunter.flush()
        self.prey.flush()
        self.cnt_rolls = 0

    def run_simulation(self):
        print("Start MC simulation....")
        while(self.hunter_nmb_wins + self.prey_nmb_wins < self.num_games):
            self.run_single()
        
        self.prey_stats_win = Counter({k: v/self.num_games \
                               for k, v in self.prey_stats_win.items()})
        self.hunter_stats_win = Counter({k: v/self.num_games \
                               for k, v in self.hunter_stats_win.items()})

        self.stats_end_game = self.prey_stats_win + self.hunter_stats_win
        print("MC simulation finished!")
        

    def run_single(self):
        choice = self.get_random_choice
#        print(choice)

        if choice in self.hunter.moves:
            if choice + self.hunter.current_pos >= self.prey.current_pos:
                self.hunter_nmb_wins += 1
                self.hunter_stats_win[self.cnt_rolls] += 1
                self.flush_players()
            else:
                self.hunter.current_pos += choice
        elif choice in self.prey.moves:
           if choice + self.prey.current_pos >= self.prey.end_limit:
                self.prey_nmb_wins += 1
                self.prey_stats_win[self.cnt_rolls] += 1
                self.flush_players()
           else:
                self.prey.current_pos += choice 

    @property
    def get_random_choice(self):
        self.cnt_rolls += 1
        return random.randint(1, 6)


if __name__ == "__main__":

    hunter_moves = [5, 6]
    prey_moves = [1, 2, 3, 4]

    ###################################################################
    hunter_start_position = 1
    prey_start_position = 7
    prey_end_position = 12
    ###################################################################
    
    hunter = PlayerMC("Hunter",
                    current_pos = hunter_start_position-1,
                    moves = hunter_moves)

    prey = PlayerMC("Prey",
                  current_pos = prey_start_position-1, 
                  moves = prey_moves, 
                  end_limit = prey_end_position-1)

    game = MCGameSimulator(hunter, prey, num_simulation_games=100000)
    game.run_simulation()


