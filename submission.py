from util import manhattanDistance
from game import Directions
import random
import util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState: GameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |gameState| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        gameState.getLegalActions(agentIndex):
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game


        The GameState class is defined in pacman.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
        """
        The evaluation function takes in the current GameState (defined in pacman.py)
        and a proposed action and returns a rough estimate of the resulting successor
        GameState's value.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    min_ghostDistance = float('inf')
    currentScore = currentGameState.getScore()

    for ghost in currentGameState.getGhostPositions():
        currentDistance = manhattanDistance(ghost, currentGameState.getPacmanPosition())
        if currentDistance < min_ghostDistance: min_ghostDistance = currentDistance

    return (currentScore - 100 * (1.0 / min_ghostDistance))
    """
    return currentGameState.getScore()
    """


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self.__numMovimientos = 0

    def getAction(self, gameState: GameState) -> str:
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Don't forget to limit the search depth using self.depth. Also, avoid modifying
          self.depth directly (e.g., when implementing depth-limited search) since it
          is a member variable that should stay fixed throughout runtime.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.getScore():
            Returns the score corresponding to the current state of the game

          gameState.isWin():
            Returns True if it's a winning state

          gameState.isLose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue

        """

        self.__numMovimientos += 1
        accionesPosibles = gameState.getLegalActions(0)

        if not accionesPosibles: # Si no hay acciones legales, paramos
            return Directions.STOP

        mejorAccion = None
        mejorValor = float('-inf')

        for accion in accionesPosibles:
            estadoSucesor = gameState.generateSuccessor(0, accion)

            valorAccion = self.minimax(estadoSucesor, agente = 1, profundidad = 0)

            if valorAccion > mejorValor: # Nos quedamos con la acción de mayor valor
                mejorValor = valorAccion
                mejorAccion = accion

        print(f"Movimiento nº {self.__numMovimientos} | "
              f"Acción: {mejorAccion} | "
              f"Valor: {round(mejorValor, 2)} | "
              f"Score actual: {gameState.getScore()}")

        estadoFinal = gameState.generateSuccessor(0, mejorAccion)

        if estadoFinal.isWin() or estadoFinal.isLose() or not estadoFinal.getLegalActions(0):
            if estadoFinal.isWin():
                resultado = "V (Victoria)"
            elif estadoFinal.isLose():
                resultado = "D (Derrota)"
            else:
                resultado = "E (Ejecutándose)"

            print(f"\n************ RESULTADOS PARA LA TABLA DE ANÁLISIS ************\n"
                  f" Score: {gameState.getScore()}\n"
                  f" Movimientos: {self.__numMovimientos}\n"
                  f" Estado: {resultado}\n")

        return mejorAccion

    def minimax(self, gameState, agente, profundidad):
        totalAgentes = gameState.getNumAgents()

        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        if profundidad == self.depth:
            return self.evaluationFunction(gameState)

        accionesLegales = gameState.getLegalActions(agente)

        if not accionesLegales:
            return self.evaluationFunction(gameState)

        siguienteAgente = (agente + 1) % totalAgentes
        siguienteProfundidad = profundidad + (1 if siguienteAgente == 0 else 0)

        # NODO MAX:
        if agente == 0:
            valorMax = float('-inf')

            for accion in accionesLegales:
                sucesor = gameState.generateSuccessor(agente, accion)
                valorHijo = self.minimax(sucesor, siguienteAgente, siguienteProfundidad)

                if valorHijo > valorMax:
                    valorMax = valorHijo

            return valorMax

        # NODO MIN:
        else:
            valorMin = float('+inf')

            for accion in accionesLegales:
                sucesor = gameState.generateSuccessor(agente, accion)
                valorHijo = self.minimax(sucesor, siguienteAgente, siguienteProfundidad)

                if valorHijo < valorMin:
                    valorMin = valorHijo

            return valorMin


######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (problem 2)
      You may reference the pseudocode for Alpha-Beta pruning here:
      en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self.__numMovimientos = 0

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        self.__numMovimientos += 1
        accionesPosibles = gameState.getLegalActions(0)

        if not accionesPosibles: # Si no hay acciones legales, paramos
            return Directions.STOP

        mejorAccion = None
        mejorValor = float('-inf')

        alfa = float('-inf')
        beta = float('+inf')

        for accion in accionesPosibles:
            estadoSucesor = gameState.generateSuccessor(0, accion)

            valorAccion = self.alphabeta(estadoSucesor, agente = 1, profundidad = 0, alfa = alfa, beta = beta)

            if valorAccion > mejorValor: # Nos quedamos con la acción de mayor valor
                mejorValor = valorAccion
                mejorAccion = accion

            if mejorValor > alfa: # Nos quedamos con el mejor valor para alfa
                alfa = mejorValor

        print(f"Movimiento nº {self.__numMovimientos} | "
              f"Acción: {mejorAccion} | "
              f"Valor: {round(mejorValor, 2)} | "
              f"Score actual: {gameState.getScore()}")

        estadoFinal = gameState.generateSuccessor(0, mejorAccion)

        if estadoFinal.isWin() or estadoFinal.isLose() or not estadoFinal.getLegalActions(0):
            if estadoFinal.isWin():
                resultado = "V (Victoria)"
            elif estadoFinal.isLose():
                resultado = "D (Derrota)"
            else:
                resultado = "E (Ejecutándose)"

            print(f"\n************ RESULTADOS PARA LA TABLA DE ANÁLISIS ************\n"
                  f" Score: {gameState.getScore()}\n"
                  f" Movimientos: {self.__numMovimientos}\n"
                  f" Estado: {resultado}\n")

        return mejorAccion

    def alphabeta(self, gameState, agente, profundidad, alfa, beta):
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        if profundidad == self.depth:
            return self.evaluationFunction(gameState)

        accionesLegales = gameState.getLegalActions(agente)
        if not accionesLegales:
            return self.evaluationFunction(gameState)

        totalAgentes = gameState.getNumAgents()

        # NODO MAX:
        if agente == 0:
            valorMax = float('-inf')

            for accion in accionesLegales:
                sucesor = gameState.generateSuccessor(agente, accion)
                valorHijo = self.alphabeta(sucesor, 1, profundidad, alfa, beta) # Agente 1: fantasmas (MIN)

                if valorHijo > valorMax:
                    valorMax = valorHijo

                if valorMax > beta: # Si el valor ya supera lo que MIN permitiría, no tiene sentido seguir explorando este nodo MAX y poda
                    return valorMax

                if valorMax > alfa: # Así MAX siempre tiene el valor máximo
                    alfa = valorMax

            return valorMax

        # NODO MIN:
        else:
            valorMin = float('+inf')

            for accion in accionesLegales:
                sucesor = gameState.generateSuccessor(agente, accion)

                if agente + 1 < totalAgentes: # Si quedan más fantasmas por jugar en esta ronda
                    valorHijo = self.alphabeta(sucesor, agente+1, profundidad, alfa, beta)
                else: # si todos los fantasmas han jugado
                    valorHijo = self.alphabeta(sucesor, 0, profundidad + 1, alfa, beta)  # Agente 0: pacman (MAX), incrementa la profundidad

                if valorHijo < valorMin:
                    valorMin = valorHijo

                if valorMin < alfa:  # Si el valor ya es menor que MAX, no tiene sentido seguir explorando este nodo MIN y poda
                    return valorMin

                if valorMin < beta: # Así MIN siempre tiene el valor mínimo
                    beta = valorMin

            return valorMin


######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self.__numMovimientos = 0

    def getAction(self, gameState: GameState) -> str:
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        self.__numMovimientos += 1
        accionesPosibles = gameState.getLegalActions(0)

        if not accionesPosibles: # Si no hay acciones legales, paramos
            return Directions.STOP

        mejorAccion = None
        mejorValor = float('-inf')

        for accion in accionesPosibles:
            estadoSucesor = gameState.generateSuccessor(0, accion)

            valorAccion = self.expectimax(estadoSucesor, agente = 1, profundidad = 0)

            if valorAccion > mejorValor: # Nos quedamos con la acción de mayor valor
                mejorValor = valorAccion
                mejorAccion = accion

        print(f"Movimiento nº {self.__numMovimientos} | "
              f"Acción: {mejorAccion} | "
              f"Valor: {round(mejorValor, 2)} | "
              f"Score actual: {gameState.getScore()}")

        estadoFinal = gameState.generateSuccessor(0, mejorAccion)

        if estadoFinal.isWin() or estadoFinal.isLose() or not estadoFinal.getLegalActions(0):
            if estadoFinal.isWin():
                resultado = "V (Victoria)"
            elif estadoFinal.isLose():
                resultado = "D (Derrota)"
            else:
                resultado = "E (Ejecutándose)"

            print(f"\n************ RESULTADOS PARA LA TABLA DE ANÁLISIS ************\n"
                  f"  Score      : {gameState.getScore()}\n"
                  f"  Movimientos: {self.__numMovimientos}\n"
                  f"  Estado     : {resultado}\n")

        return mejorAccion

    def expectimax(self, gameState, agente, profundidad):
        totalAgentes = gameState.getNumAgents()

        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        if profundidad == self.depth:
            return self.evaluationFunction(gameState)

        accionesLegales = gameState.getLegalActions(agente)
        if not accionesLegales:
            return self.evaluationFunction(gameState)

        siguienteAgente = (agente + 1) % totalAgentes
        siguienteProfundidad = profundidad + (1 if siguienteAgente == 0 else 0)

        # NODO MAX:
        if agente == 0:
            valorMax = float('-inf')

            for accion in accionesLegales:
                sucesor = gameState.generateSuccessor(agente, accion)
                valorHijo = self.expectimax(sucesor, siguienteAgente, siguienteProfundidad)

                # Nos quedamos con el mayor valor encontrado
                if valorHijo > valorMax:
                    valorMax = valorHijo

            return valorMax

        # NODO AZAR:
        else:
            sumaValores = 0.0

            for accion in accionesLegales:
                sucesor = gameState.generateSuccessor(agente, accion)
                valorHijo = self.expectimax(sucesor, siguienteAgente, siguienteProfundidad)
                sumaValores += valorHijo

            return sumaValores / len(accionesLegales) # Media aritmética de todos los sucesores


######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
    puntuacion = currentGameState.getScore()
    posicionPacman = currentGameState.getPacmanPosition()
    posicionesFantasmas = currentGameState.getGhostPositions()

    if not posicionesFantasmas:
        return puntuacion

    distanciaMinima = min(manhattanDistance(posicionPacman, posFantasma) for posFantasma in posicionesFantasmas) # Distancia al fantasma más cercano

    if distanciaMinima > 0: # Si el fantasma no está en la misma casilla
        puntuacion -= 100.0 * (1.0 / distanciaMinima)
    else: # Si el fantasma está en la misma casilla
        puntuacion -= 10000

    return puntuacion

# Abbreviation
better = betterEvaluationFunction