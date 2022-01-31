#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: evgeniy
"""
from HunterMooseCode import Player, GameSimulator
from HunterMooseCodeMonteCarlo import PlayerMC, MCGameSimulator



###################################################################
####   Probability Tree Diagrams simulation
##################################################################  
def probability_tree_diagram_simulation(hunter_start,
                                        hunter_moves,
                                        prey_start,
                                        prey_moves,
                                        prey_end):
    
    print("========-----------===========------------========")

    hunter = Player("Hunter", current_pos=hunter_start-1,
                    moves=hunter_moves)
    prey = Player("Prey", current_pos=prey_start-1, 
                    moves=prey_moves,
                    end_limit=prey_end-1)
    
            
    game = GameSimulator(hunter, prey)  

    game.run_simulation()
    game.print_stats()
    
    print("========-----------===========------------========")




###################################################################
####   Monte Carlo simulation
##################################################################      
def monte_carlo_simulation(hunter_start,
                            hunter_moves,
                            prey_start,
                            prey_moves,
                            prey_end,
                            num_simulation_games):

    print("========-----------===========------------========")
    
    hunter = PlayerMC("Hunter",
                    current_pos = hunter_start-1,
                    moves = hunter_moves)

    prey = PlayerMC("Prey",
                  current_pos = prey_start-1, 
                  moves = prey_moves, 
                  end_limit = prey_end-1)

    game_mc = MCGameSimulator(hunter, prey, 
                              num_simulation_games=num_simulation_games)
    print("Number of simulation games: %d" % num_simulation_games)
    game_mc.run_simulation()
    
    game_mc.print_stats()
    
    print("========-----------===========------------========")

if __name__ == "__main__":

    hunter_moves = [5, 6]
    prey_moves = [1, 2, 3, 4]

    ###################################################################
    hunter_start_position = 1
    prey_start_position = 9
    prey_end_position = 15
    ###################################################################
  
    probability_tree_diagram_simulation(hunter_start=hunter_start_position,
                                        hunter_moves=hunter_moves,
                                        prey_start=prey_start_position,
                                        prey_moves=prey_moves,
                                        prey_end=prey_end_position)

    number_simulation_games = 1000000
    monte_carlo_simulation(hunter_start=hunter_start_position,
                            hunter_moves=hunter_moves,
                            prey_start=prey_start_position,
                            prey_moves=prey_moves,
                            prey_end=prey_end_position,
                            num_simulation_games=number_simulation_games)

