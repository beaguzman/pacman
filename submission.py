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

        # Contamos este movimiento para la estadística final
        self.__numMovimientos += 1

        # Recuperamos todas las acciones que puede hacer pacman ahora mismo
        accionesPosibles = gameState.getLegalActions(self.index)

        # Si por algún motivo no hay acciones legales, nos detenemos
        if not accionesPosibles:
            return Directions.STOP

        # Para cada acción generamos el estado sucesor y lo evaluamos con minimax. Guardamos la acción que da el valor más alto.
        mejorAccion = None
        mejorValor = float('-inf')  # Empezamos con el peor caso posible

        for accion in accionesPosibles:
            # Generamos el tablero resultante de que pacman haga esta acción
            estadoSucesor = gameState.generateSuccessor(self.index, accion)

            # Ahora le toca al agente 1 (primer fantasma), profundidad 0 porque todavía no hemos completado ningún nivel completo
            valorAccion = self.minimax(estadoSucesor, agente =self.index+1, profundidad = 0)

            # Nos quedamos con la acción de mayor valor
            if valorAccion > mejorValor:
                mejorValor = valorAccion
                mejorAccion = accion

        # Print de seguimiento por cada movimiento
        print(f"Movimiento nº {self.__numMovimientos} | "
              f"Acción: {mejorAccion} | "
              f"Valor: {round(mejorValor, 2)} | "
              f"Score actual: {gameState.getScore()}")

        estadoFinal = gameState.generateSuccessor(self.index, mejorAccion)

        if estadoFinal.isWin() or estadoFinal.isLose() or not estadoFinal.getLegalActions(0):
            if estadoFinal.isWin():
                resultado = "V (Victoria)"
            elif estadoFinal.isLose():
                resultado = "D (Derrota)"
            else:
                resultado = "E (Ejecutándose)"

            print(f"\n************ RESULTADOS PARA LA TABLA DE ANÁLISIS ************\n"
                  f"  Score: {estadoFinal.getScore()}\n"
                  f"  Movimientos: {self.__numMovimientos}\n"
                  f"  Estado: {resultado}\n")

        return mejorAccion

    def minimax(self, gameState, agente, profundidad):
        totalAgentes = gameState.getNumAgents()

        # --- CASOS BASE --
        # Pacman ganó o perdió
        if gameState.isWin() or gameState.isLose():
            # Devolvemos la puntuación real, no la heurística, porque ya sabemos el resultado definitivo
            return gameState.getScore()

        # Límite de profundidad alcanzado: usamos la heurística
        if profundidad == self.depth:
            return self.evaluationFunction(gameState)

        # Acciones disponibles para el agente actual
        accionesLegales = gameState.getLegalActions(agente)

        # Sin acciones posibles: el agente está bloqueado
        if not accionesLegales:
            return self.evaluationFunction(gameState)

        # --- CÁLCULO DEL SIGUIENTE TURNO ---
        # Los agentes rotan: 0 → 1 → 2 → ... → (N-1) → 0 → 1 → ...
        # Cuando volvemos al agente 0 significa que una ronda entera ha pasado, por eso incrementamos la profundidad.
        siguienteAgente = (agente + 1) % totalAgentes
        siguienteProfundidad = profundidad + (1 if siguienteAgente == self.index else 0)

        # NODO MAX: turno de pacman (agente 0)
        if agente == self.index:
            valorMax = float('-inf')

            for accion in accionesLegales:
                sucesor = gameState.generateSuccessor(agente, accion)
                valorHijo = self.minimax(sucesor, siguienteAgente, siguienteProfundidad)

                if valorHijo > valorMax:
                    valorMax = valorHijo

            return valorMax

        # NODO MIN: turno de un fantasma (agente 1)
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
        self.__numMovimientos = 0  # Contador de movimientos realizados

    def getAction(self, gameState):
        # Contamos este movimiento para la estadística final
        self.__numMovimientos += 1

        # Acciones legales de pacman en el estado actual
        accionesPosibles = gameState.getLegalActions(self.index)

        # Si no hay acciones disponibles, pacman se detiene
        if not accionesPosibles:
            return Directions.STOP

        mejorAccion = None
        mejorValor = float('-inf')

        # Inicializamos alfa y beta para la raíz del árbol.
        # alfa = -inf ; MAX aún no tiene ninguna garantía.
        # beta = +inf ; MIN aún no tiene ninguna garantía.
        alfa = float('-inf')
        beta = float('+inf')

        for accion in accionesPosibles:
            # Generamos el estado resultante de que pacman haga esta acción
            estadoSucesor = gameState.generateSuccessor(self.index, accion)

            # Ahora le toca al agente 1 (primer fantasma), profundidad 0
            valorAccion = self.alphabeta(estadoSucesor, agente = self.index+1, profundidad = 0, alfa = alfa, beta = beta)

            # Actualizamos la mejor acción encontrada hasta ahora
            if valorAccion > mejorValor:
                mejorValor = valorAccion
                mejorAccion = accion

            # Actualizamos alfa en la raíz: MAX ya puede garantizarse mejorValor
            # Esto permite podar ramas en llamadas siguientes de este mismo bucle
            if mejorValor > alfa:
                alfa = mejorValor

        # Print de seguimiento por cada movimiento
        print(f"Movimiento nº {self.__numMovimientos} | "
              f"Acción: {mejorAccion} | "
              f"Valor: {round(mejorValor, 2)} | "
              f"Score actual: {gameState.getScore()}")

        # Comprobamos si el estado siguiente es terminal para mostrar el resumen de la tabla solo una vez, al acabar la partida
        estadoFinal = gameState.generateSuccessor(self.index, mejorAccion)

        if estadoFinal.isWin() or estadoFinal.isLose() or not estadoFinal.getLegalActions(self.index):
            if estadoFinal.isWin():
                resultado = "V (Victoria)"
            elif estadoFinal.isLose():
                resultado = "D (Derrota)"
            else:
                resultado = "E (Ejecutándose)"

            print(f"\n************ RESULTADOS PARA LA TABLA DE ANÁLISIS ************\n"
                  f"  Score: {estadoFinal.getScore()}\n"
                  f"  Movimientos: {self.__numMovimientos}\n"
                  f"  Estado: {resultado}\n")

        return mejorAccion

    def alphabeta(self, gameState, agente, profundidad, alfa, beta):
        totalAgentes = gameState.getNumAgents()

        # --- CASOS BASE ---
        # Pacman ganó o perdió
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        # Profundidad máxima alcanzada: usamos la heurística
        if profundidad == self.depth:
            return self.evaluationFunction(gameState)

        # Acciones disponibles para el agente actual
        accionesLegales = gameState.getLegalActions(agente)

        # Sin acciones posibles: el agente está bloqueado
        if not accionesLegales:
            return self.evaluationFunction(gameState)

        # --- CÁLCULO DEL SIGUIENTE TURNO ---
        # Igual que en MiniMax y ExpectiMax: rotamos agentes y subimos profundidad cuando volvemos al agente 0.
        siguienteAgente = (agente + 1) % totalAgentes
        siguienteProfundidad = profundidad + (1 if siguienteAgente == self.index else 0)


        # NODO MAX: turno de pacman (agente 0)
        if agente == self.index:
            valorMax = float('-inf')

            for accion in accionesLegales:
                sucesor = gameState.generateSuccessor(agente, accion)
                valorHijo = self.alphabeta(sucesor, siguienteAgente, siguienteProfundidad, alfa, beta)

                if valorHijo > valorMax:
                    valorMax = valorHijo

                # Poda beta: si el valor ya supera lo que MIN permitiría,
                # no tiene sentido seguir explorando este nodo MAX
                if valorMax >= beta:
                    return valorMax

                # Actualizamos alfa: MAX ya puede garantizarse valorMax
                if valorMax > alfa:
                    alfa = valorMax

            return valorMax

        # NODO MIN: turno de un fantasma (agente 1)
        else:
            valorMin = float('+inf')

            for accion in accionesLegales:
                sucesor = gameState.generateSuccessor(agente, accion)
                valorHijo=self.alphabeta(sucesor,siguienteAgente, siguienteProfundidad, alfa, beta)

                if valorHijo < valorMin:
                    valorMin = valorHijo

                # Poda alfa: si el valor ya es menor que lo que MAX garantiza, no tiene sentido seguir explorando este nodo MIN
                if valorMin <= alfa:
                    return valorMin

                # Actualizamos beta: MIN ya puede garantizarse valorMin
                if valorMin < beta:
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

        # Contamos este movimiento para la estadística final
        self.__numMovimientos += 1

        # Acciones legales de pacman en el estado actual
        accionesPosibles = gameState.getLegalActions(self.index)

        # Si no hay acciones disponibles, pacman se detiene
        if not accionesPosibles:
            return Directions.STOP

        mejorAccion = None
        mejorValor = float('-inf')  # Empezamos con el peor valor posible

        for accion in accionesPosibles:
            # Generamos el estado resultante de que pacman haga esta acción
            estadoSucesor = gameState.generateSuccessor(self.index, accion)

            # Ahora le toca al agente 1 (primer fantasma), profundidad 0.
            # Al ser fantasma, llamamos al nodo de azar directamente.
            valorAccion = self.expectimax(estadoSucesor, agente = self.index+1, profundidad = 0)

            # Nos quedamos con la acción de mayor valor esperado
            if valorAccion > mejorValor:
                mejorValor = valorAccion
                mejorAccion = accion

        # Print de seguimiento por cada movimiento
        print(f"Movimiento nº {self.__numMovimientos} | "
              f"Acción: {mejorAccion} | "
              f"Valor: {round(mejorValor, 2)} | "
              f"Score actual: {gameState.getScore()}")

        # Comprobamos si el estado siguiente es terminal para mostrar el resumen de la tabla solo una vez, al acabar la partida
        estadoFinal = gameState.generateSuccessor(self.index, mejorAccion)

        if estadoFinal.isWin() or estadoFinal.isLose() or not estadoFinal.getLegalActions(self.index):
            if estadoFinal.isWin():
                resultado = "V (Victoria)"
            elif estadoFinal.isLose():
                resultado = "D (Derrota)"
            else:
                resultado = "E (Ejecutándose)"

            print(f"\n************ RESULTADOS PARA LA TABLA DE ANÁLISIS ************\n"
                  f"  Score: {estadoFinal.getScore()}\n"
                  f"  Movimientos: {self.__numMovimientos}\n"
                  f"  Estado: {resultado}\n")

        return mejorAccion

    def expectimax(self, gameState, agente, profundidad):
        totalAgentes = gameState.getNumAgents()

        # --- CASOS BASE ---
        # Pacman ganó o perdió
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        # Profundidad máxima alcanzada: usamos la heurística
        if profundidad == self.depth:
            return self.evaluationFunction(gameState)

        # Acciones disponibles para el agente actual
        accionesLegales = gameState.getLegalActions(agente)

        # Sin acciones posibles: el agente está bloqueado
        if not accionesLegales:
            return self.evaluationFunction(gameState)

        # --- CÁLCULO DEL SIGUIENTE TURNO ---
        # Los agentes rotan igual que en MiniMax.
        # Cuando volvemos al agente 0, incrementamos la profundidad.
        siguienteAgente = (agente + 1) % totalAgentes
        siguienteProfundidad = profundidad + (1 if siguienteAgente == self.index else 0)

        # NODO MAX: turno de pacman (agente 0)
        if agente == self.index:
            valorMax = float('-inf')

            for accion in accionesLegales:
                sucesor = gameState.generateSuccessor(agente, accion)
                valorHijo = self.expectimax(sucesor, siguienteAgente, siguienteProfundidad)

                # Nos quedamos con el mayor valor encontrado
                if valorHijo > valorMax:
                    valorMax = valorHijo

            return valorMax

        # NODO AZAR: turno de un fantasma (agente 1)
        else:
            # A diferencia de MiniMax, aquí no buscamos el mínimo.
            # Sumamos todos los valores y devolvemos la media, porque asumimos que el fantasma elige aleatoriamente con igual probabilidad cada acción disponible.
            sumaValores = 0.0

            for accion in accionesLegales:
                sucesor = gameState.generateSuccessor(agente, accion)
                valorHijo = self.expectimax(sucesor, siguienteAgente, siguienteProfundidad)
                sumaValores += valorHijo

            # Valor esperado = media aritmética de todos los sucesores
            return sumaValores / len(accionesLegales)


######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
    # Partimos de la puntuación actual del juego como valor base
    puntuacion = currentGameState.getScore()

    # Obtenemos la posición actual de pacman
    posicionPacman = currentGameState.getPacmanPosition()

    # Obtenemos las posiciones de todos los fantasmas
    posicionesFantasmas = currentGameState.getGhostPositions()

    # Solo penalizamos si hay fantasmas en el tablero
    if not posicionesFantasmas:
        return puntuacion

    # Calculamos la distancia Manhattan de pacman a cada fantasma y nos quedamos con la del más cercano
    distanciaMinima = min(manhattanDistance(posicionPacman, posFantasma) for posFantasma in posicionesFantasmas)

    # Aplicamos la penalización según la fórmula del enunciado: cuanto más cerca está el fantasma, mayor es la penalización.
    # Si el fantasma está en la misma casilla, penalizamos con un valor muy alto para evitar ese estado a toda costa.
    if distanciaMinima > 0:
        puntuacion -= 100.0 * (1.0 / distanciaMinima)
    else:
        puntuacion -= 10000

    return puntuacion


# Abbreviation
better = betterEvaluationFunction