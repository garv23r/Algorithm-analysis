# :mortar_board: Algorithm Comparison and Implementation Project

## :pushpin: Project Summary

This project, developed with the contributions of **Arshnoor Kaur** and **Zain Siddhiqi**, provides a comprehensive comparison of the Dijkstra and Bellman-Ford algorithms, exploring their performance under various conditions. The study is divided into several sections, each focusing on different aspects of these algorithms, including their efficiency, accuracy, and applicability in various scenarios. Additionally, the project delves into the A* algorithm, examining its advantages over Dijkstra's algorithm and implementing it using the Adapter pattern.

## :sunglasses: Tech Stack

- **Programming Language:** Python
- **Libraries and Frameworks:**
  - **NumPy:** For numerical operations and handling large datasets.
  - **Matplotlib:** For visualizing the results of the experiments with graphs and charts.
  - **Priority Queue:** Custom implementation to manage node priorities in A* algorithm.

## :ledger: Table of Contents

1. [List of Figures](#list-of-figures)
2. [Section 1](#section-1)
    - [Part 1.3](#part-13)
        - [Experiment 1: Comparing the Dijkstra and Bellman-Ford Algorithms on the basis of the number of runs](#experiment-1-comparing-the-dijkstra-and-bellman-ford-algorithms-on-the-basis-of-the-number-of-runs)
        - [Experiment 2: Comparing the Dijkstra and Bellman-Ford Algorithms on the basis of the number of maximum allowed relaxations (k)](#experiment-2-comparing-the-dijkstra-and-bellman-ford-algorithms-on-the-basis-of-the-number-of-maximum-allowed-relaxations-k)
        - [Experiment 3: Comparing the Dijkstra and Bellman-Ford Algorithms on the basis of the size of the graph (number of nodes)](#experiment-3-comparing-the-dijkstra-and-bellman-ford-algorithms-on-the-basis-of-the-size-of-the-graph-number-of-nodes)
        - [Experiment 4: Comparing the Dijkstra and Bellman-Ford Algorithms on the basis of the density of the graph](#experiment-4-comparing-the-dijkstra-and-bellman-ford-algorithms-on-the-basis-of-the-density-of-the-graph)
    - [Time Complexity Analysis](#time-complexity-analysis)
    - [Accuracy Analysis](#accuracy-analysis)
3. [Section 2](#section-2)
4. [Section 3](#section-3)
    - [Part 3.2](#part-32)
        - [What Issues with Dijkstra’s Algorithm is A* Trying to Address?](#what-issues-with-dijkstras-algorithm-is-a-trying-to-address)
        - [Empirical Testing Methodology](#empirical-testing-methodology)
        - [Comparison with Arbitrary Heuristic Functions](#comparison-with-arbitrary-heuristic-functions)
        - [Applications of A* over Dijkstra’s](#applications-of-a-over-dijkstras)
5. [Section 4](#section-4)
6. [Section 5](#section-5)
    - [A* Algorithm Implementation as an “Adapter”](#a-algorithm-implementation-as-an-adapter)
    - [Adapter Pattern Overview](#adapter-pattern-overview)
    - [Implementation Steps](#implementation-steps)
        - [Creating the AStar Adapter](#creating-the-astar-adapter)
        - [Priority Queue](#priority-queue)
        - [Calculation of Total Cost](#calculation-of-total-cost)
        - [Priority Queue Insertion](#priority-queue-insertion)
        - [Path Retrieval and Backtracking](#path-retrieval-and-backtracking)
    - [Robustness and Flexibility](#robustness-and-flexibility)
    - [Conclusion](#conclusion)
    - [Design Principles and Patterns](#design-principles-and-patterns)
        - [Encapsulation](#encapsulation)
        - [Inheritance](#inheritance)
        - [Polymorphism](#polymorphism)
        - [Abstraction](#abstraction)
    - [Class Hierarchy](#class-hierarchy)
    - [Attributes and Methods](#attributes-and-methods)
    - [Inheritance Relationships](#inheritance-relationships)
    - [Abstraction and Encapsulation](#abstraction-and-encapsulation)
    - [Modified Node Representation](#modified-node-representation)
        - [Node Representation](#node-representation)
        - [Modify Existing Classes](#modify-existing-classes)
    - [Node Attributes](#node-attributes)
    - [Heuristics and Custom Data](#heuristics-and-custom-data)
    - [Edge Representation](#edge-representation)
    - [Overall Benefits](#overall-benefits)
    - [Other Types of Graph Implementation](#other-types-of-graph-implementation)
        - [Finite Graphs](#finite-graphs)
        - [Infinite Graphs](#infinite-graphs)
        - [Weighted Graphs](#weighted-graphs)
        - [Directed Graphs (Digraphs)](#directed-graphs-digraphs)
        - [Acyclic Graphs (DAGs)](#acyclic-graphs-dags)
        - [Random Graphs](#random-graphs)

## :dart: Key Sections and Experiments

### Comparison of Dijkstra and Bellman-Ford Algorithms

- **Experiment 1:** Evaluates both algorithms based on the number of runs required to find the shortest path, providing insights into their performance under repeated operations.
- **Experiment 2:** Analyzes the effect of the maximum allowed relaxations (k) on the algorithms, highlighting their adaptability to constraints.
- **Experiment 3:** Assesses how the size of the graph (in terms of the number of nodes) impacts the algorithms' efficiency, offering a perspective on scalability.
- **Experiment 4:** Compares the algorithms based on graph density, showing how they handle different levels of graph complexity.

### Time and Accuracy Analysis

- A thorough examination of the time complexity of both algorithms is provided, alongside an analysis of their accuracy in finding the shortest paths.

### Exploration of the A* Algorithm

- **What Issues with Dijkstra’s Algorithm is A* Trying to Address?**  
  Discusses the limitations of Dijkstra's algorithm that A* aims to solve, particularly in terms of efficiency and path optimality.
- **Empirical Testing Methodology:**  
  Details the testing approach used to evaluate A*, ensuring rigorous comparison with Dijkstra’s algorithm.
- **Comparison with Arbitrary Heuristic Functions:**  
  Examines how A* performs with different heuristic functions, showcasing its flexibility and power.

### Implementation of the A* Algorithm Using the Adapter Pattern

- The project demonstrates how the A* algorithm can be implemented as an "Adapter," allowing it to be integrated into existing systems seamlessly.
- **Key Implementation Steps:**
  - **Creating the AStar Adapter:**  
    A detailed guide on constructing the adapter for A*.
  - **Priority Queue Management:**  
    Describes the role of the priority queue in managing node traversal priorities.
  - **Total Cost Calculation and Path Retrieval:**  
    Explains the process of calculating costs and retrieving the optimal path using A*.

### Design Principles and Patterns

- **Encapsulation, Inheritance, Polymorphism, and Abstraction:**  
  The project showcases the application of these design principles in the context of algorithm implementation.
- **Class Hierarchy and Attributes:**  
  Provides a breakdown of the class structure, attributes, and methods used in the implementation, ensuring a clear understanding of the codebase.

### Graph Representations

The project discusses various types of graphs (finite, infinite, weighted, directed, acyclic, and random) and their implementations, demonstrating the versatility of the algorithms in different graph scenarios.

## :tada: Conclusion

The project concludes with a reflection on the robustness and flexibility of the A* algorithm when implemented using the Adapter pattern, along with the overall benefits of each algorithm in different contexts. Through detailed experiments and analyses, this study provides a thorough understanding of Dijkstra, Bellman-Ford, and A* algorithms, their strengths, weaknesses, and practical applications in solving shortest path problems.
