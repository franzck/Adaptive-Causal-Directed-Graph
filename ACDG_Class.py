import igraph as ig
from xml.etree import ElementTree as ET
import time
import pickle
import plotly.graph_objects as go
import re
import numpy as np
import heapq
import concurrent.futures

class Valve:
           
            global self
            def __init__(self, Tag) -> None:
                """
                Initializes a new Valve object with a unique identifier (Tag) and sets up its internal state.
                self.history_length: Number of time steps to track the valve's state.
                self.state_history: A list representing the valve's state over time. Initialized to 0.5 for all steps.
                self.Tag: A unique identifier for the valve.
                self.relationDict: A dictionary to store relationships with other components, categorized by type (Temperature, Pressure, etc.).
                self.kernelDict: Stores kernel weights for each carrier type—used for processing historical data.
                self.valuekernelDict: Stores different kernel weights, likely for smoothing or filtering values.
                """
                self.history_length = 60
                self.state_history = [0.5] * self.history_length #self.history_kernel(0.9,0.3,10,50) #[0] * self.history_length
                self.Tag = Tag
                self.relationDict = {"Temperature":[], "Pressure":[], "Flow":[], "Level":[]}#, "State":[]}
                self.kernelDict = {"Temperature":self.history_kernel(1,1,1,1), "Pressure":self.history_kernel(1,1,1,1), "Flow":self.history_kernel(1,1,1,1), "Level":self.history_kernel(1,1,1,1)}
                self.valuekernelDict = {"Temperature":self.history_kernel(0.1,0.99,1,30), "Pressure":self.history_kernel(0.1,0.99,1,30), "Flow":self.history_kernel(0.1,0.99,1,30), "Level":self.history_kernel(0.1,0.99,1,30)}
                
            def find_relations(self, igraph):
                """
                Identifies and stores all edges in the given graph (`igraph`) that are related to this valve,
                grouped by carrier type.

                Parameters:
                    igraph (igraph.Graph): A directed graph where vertices have a "Tag" attribute and edges have a "carrier" attribute.

                Side Effects:
                    - Updates `self.relationDict`, which maps each carrier type to a list of edge indices
                    that are connected to this valve and match the carrier type.
                """
                for carriertype in self.relationDict:
                    #Find all childs of the valve with the carriertype carrier
                    for child in igraph.successors(igraph.vs["Tag"].index(self.Tag)):
                        for relation in igraph.es.select(_source=child):
                            #Find all edges with the carrier Type and add them to the relationDict
                            if relation["carrier"] == carriertype:
                                self.relationDict[carriertype].append(relation.index)
                        for relation in igraph.es.select(_target=child):
                            if relation["carrier"] == carriertype:
                                self.relationDict[carriertype].append(relation.index)
  
            def apply_weights(self, igraph):
                """
                Applies computed weights to the valve's related edges in the graph.

                The weight for each edge is calculated using:
                    - `kernelDict`: A kernel function representing the importance of each time step.
                    - `valuekernelDict`: A mapping from discretized state values to weight contributions.
                    - `state_history`: A list of historical state values for the valve.

                Parameters:
                    igraph (igraph.Graph): A graph where edges can be annotated with "weight" and "Actors".

                Side Effects:
                    - Updates each edge in `self.relationDict` with a computed "weight".
                    - Annotates each edge with the valve's tag under the "Actors" attribute.
                """
                for carrier in self.relationDict:
                    temp_history = self.state_history
                    temp_value_history = [0.5] * self.history_length
                    counter = 0
                    for hist_val in temp_history:
                        hist_val = int(np.round(hist_val*self.history_length))
                        if hist_val > self.history_length-1:
                            hist_val = self.history_length-1
                        elif hist_val < 0:
                            hist_val = 0
                        temp_value_history[counter] = self.valuekernelDict[carrier][hist_val]
                        counter += 1
                        
                    current_weight = np.sum(np.array(temp_value_history)*np.array(self.kernelDict[carrier]))/np.sum(self.kernelDict[carrier])
                    for relations in self.relationDict[carrier]:
                        igraph.es[relations]["weight"] = current_weight
                        igraph.es[relations]["Actors"] = self.Tag

            def update_state(self, in_state):
                """
                Updates the valve's state history with a new state value.

                Parameters:
                    in_state (float or int): The latest state value to be added to the history.

                Side Effects:
                    - Appends the new state to `self.state_history`.
                    - Removes the oldest state to maintain a fixed history length.
                """
                self.state_history.append(in_state)
                self.state_history.pop(0)
                
            
            def history_kernel(self, start_value, stop_value, ramp_startpoint, ramplength):
                """
                Generates a kernel vector used for weighting historical data.

                The kernel starts with a constant `start_value`, then ramps linearly to `stop_value` 
                over `ramplength` steps starting at `ramp_startpoint`, and finally holds at `stop_value` 
                for the remainder of the kernel.

                Parameters:
                - start_value (float): Initial value before the ramp begins.
                - stop_value (float): Final value after the ramp ends.
                - ramp_startpoint (int): Index in the kernel where the ramp begins.
                - ramplength (int): Number of steps over which the ramp occurs.

                Returns:
                - List[float]: A list of length `self.history_length` representing the kernel.
                """
                kernellength=self.history_length
                #Creates a kernel vector with a ramp from start_value to stop_value with length ramplength
                kernel = (
                    [start_value] * ramp_startpoint + 
                    list(np.linspace(start_value, stop_value, ramplength)) + 
                    [stop_value] * (kernellength - ramp_startpoint - ramplength)
                )
                return kernel

def flatten(nested_list):
    """
    Flattens a nested list of integers into a list of sublists, where each sublist
    contains integers from the same depth level of nesting.

    Parameters:
        nested_list (list): A list that may contain integers or further nested lists of integers.

    Returns:
        List[List[int]]: A list of lists, where each inner list contains integers from a specific depth level.
                         The first sublist contains integers from the outermost level, the second from one level deeper, and so on.

    Example:
        flatten([1, [2, [3, 4], 5], 6]) 
        → [[1, 6], [2, 5], [3, 4]]
    """
    def flatten_helper(nested, flat):
        for item in nested:
            if isinstance(item, int):
                flat[-1].append(item)
            else:
                flat.append([])
                flatten_helper(item, flat)
    result = [[]]
    flatten_helper(nested_list, result)
    return result

class ACDG_Class:
    """
    ACDG_Class is a class for creating and manipulating directed graphs with vertices and edges representing various elements and their relationships. 
    It provides methods for adding vertices, finding routes, changing edge hierarchies, generating graphs from AML files, applying rules, and evaluating alarm states.
    Attributes:
        counter (int): A counter for tracking the number of vertices.
        boolInitComplete (bool): A flag indicating whether initialization is complete.
        NumberOfElements (int): The number of elements in the graph.
        g (ig.Graph): The graph object.
    Methods:
        __init__(): Initializes the ACDG_Class object.
        add_myvertex(in_tag, in_carriertype, in_amlID, in_alarmtag, in_alarmstate, in_alarmlimits): Adds a vertex to the graph with specified attributes.
        find_routes(vertex1_id, vertex2_type): Finds all valid routes in the graph from a given vertex to vertices of a specified type.
        change_edge_hierarchieparents(): Changes the carrier of edges with carrier="hasparent" to "hashierarchieparent" if certain conditions are met.
        generate_graphfromAML(amlfilepath): Generates a graph from an AutomationML (AML) file.
        update_valves(in_valvedict): Updates the state of valves in the graph based on the provided dictionary.
        evaluate_valvestate(in_valuedict): Evaluates the state of valves in the graph based on the provided dictionary.
        modified_dijkstra(source_vertex_id, target_vertex_id, print_paths=True): Finds the shortest path between two vertices using a modified Dijkstra algorithm, working on positive and negative weighted paths in parallel.
        modified_dijkstra_WAI(source_vertex_id, target_vertex_id): Finds the shortest path between two vertices using a modified Dijkstra algorithm with additional checks.
        dfs(source_vertex_id, target_vertex_id, visited, weight): Performs a depth-first search to find the best path and weight between two vertices.
        evaluate_alarm(source_vertex_id, target_vertex_id): Evaluates the alarm state between two vertices using depth-first search.
        generate_alarmPLUT(): Generates a weight matrix for all active alarms in the graph.
        find_paths(carriercondition, carriermax): Finds paths in the graph that match specified carrier conditions and maximum lengths.
        explore_edges(source_vertex_id, carrierlist, carriermax, used_carriers=None, vertex_path=None, carrier_index=0, carrier_count=0): Explores edges in the graph to find paths that match specified carrier conditions and maximum lengths.
        print_edge_vertices(): Prints the source and target vertices of all edges in the graph.
        apply_rule(source_type, target_type, rulenumber, strength=0.99, effect_factor=1, time_constant=0, rationale=None, apply_carrier=None, searchpath=None, shortest=False, findValves=False, inverse=False): Applies a rule to the graph to add edges between vertices of specified types.
        _vertex_has_parent(vertex, parent_tag): Checks if a vertex has a certain parent.
        plot_graph3D(): Plots the graph in 3D using Plotly.
        plot_graph(): Plots the graph in 2D using Plotly.
        generate_distancematrix(): Generates a distance matrix for all active alarms in the graph.
        generate_distance(i, active_alarm_list, distance_matrix): Generates the distance between a vertex and all other active alarms.
        generate_distancematrix_multithreaded(): Generates a distance matrix for all active alarms in the graph using multithreading.
        evaluate_distance_matrix(distance_limit, update_distance=False): Evaluates the distance matrix to find clusters of alarms that can be reached from each other by a distance smaller than the distance limit.
        set_alarmstates(alarmstates_dict): Sets the alarm states of vertices in the graph based on the provided dictionary.
        """
    def __init__(self) -> None:
        """
        Initializes an instance with a directed igraph graph and tracking attributes.

        Attributes:
            counter (int): A general-purpose counter, initialized to 0.
            boolInitComplete (bool): Flag indicating whether initialization is complete.
            NumberOfElements (int): Tracks the number of elements (e.g., vertices or components).
            g (igraph.Graph): A directed graph initialized with zero vertices.
        """
        intVertexNumber = 0
        self.counter = 0
        self.boolInitComplete = False
        self.NumberOfElements = 0
        self.g = ig.Graph(intVertexNumber, directed=True)


    def add_myvertex(self, in_tag, in_carriertype, in_amlID, in_alarmtag, in_alarmstate, in_alarmlimits):
        """
        Adds a vertex to the internal graph with metadata and a color based on carrier type.

        Parameters:
            in_tag (str): Unique identifier for the vertex.
            in_carriertype (str): Type of carrier (e.g., "Temperature", "Pressure", "Flow").
            in_amlID (str): AML (AutomationML) identifier for the element.
            in_alarmtag (str): Tag associated with the alarm.
            in_alarmstate (str): Current alarm state.
            in_alarmlimits (str): Alarm limits or thresholds.

        Behavior:
            - Assigns a color to the vertex based on the carrier type.
            - Adds the vertex to the graph `self.g` with all provided attributes.
            - Sets `aktorinfluence` to `False` by default.

        Carrier-to-Color Mapping:
            - Temperature → red
            - Pressure → green
            - Flow → blue
            - Level → yellow
            - State → pink
            - Concentration (contains) → orange
            - Other → slategray
        """
        if in_carriertype == "Temperature":
            color = "red"
        elif in_carriertype == "Pressure":
            color = "green"
        elif in_carriertype == "Flow":
            color = "blue"
        elif in_carriertype == "Level":
            color = "yellow"
        elif in_carriertype== "State":
            color="pink"
        elif "Concentration" in in_carriertype:
            color="orange"
        else:
            color = "slategray"
        
        self.g.add_vertex(Tag=in_tag, Carrier=in_carriertype, AMLID=in_amlID ,AT=in_alarmtag, AS=in_alarmstate, AL=in_alarmlimits, color=color, aktorinfluence=False)


    def find_routes(self, vertex1_id, vertex2_type):
        """
        Find all valid routes in the graph from a given vertex to vertices of a specified type.

        This method searches for paths from a starting vertex (vertex1_id) to all vertices of a specified type (vertex2_type).
        It filters the paths to ensure they follow a specific edge order and vertex type criteria.

        Args:
            vertex1_id (int): The ID of the starting vertex.
            vertex2_type (str): The type of the target vertices.

        Returns:
            list of tuple: A list of tuples, each containing a pair of vertex IDs that form a valid route.
        """
        # Find all vertices of type vertex2_type
        vertices2_ids = [v.index for v in self.g.vs if v['Carrier'] == vertex2_type]
        valid_routes = []
        # Define the edge types in the required order
        edge_order = ['hasparent', 'Product', 'haschild']
        # Loop through all potential end vertices
        for vertex2_id in vertices2_ids:
            # Find all paths from vertex1 to vertex2
            all_paths = self.g.get_all_simple_paths(vertex1_id, to=vertex2_id)
            # Loop through all the paths
            for path in all_paths:
                # Check if the path has the correct length (4 vertices -> 3 edges)
                if len(path) != 4:
                    continue
                # Retrieve the edges in the path
                edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]                
                # Check that the edges have the correct types
                edge_types = [self.g.es[self.g.get_eid(*edge_pair)]["Carrier"] for edge_pair in edges]
                if edge_types == edge_order:
                    # Retrieve the vertices
                    vertices = [self.g.vs[vert_id]["Carrier"] for vert_id in path]                    
                    # Check that the vertices have the correct types
                    if vertices[0] == vertices[3] == 'Temperature' and vertices[1] == 'Any Type':
                        valid_routes.append((path[0], path[3]))  # Add the pair of vertices to the valid_routes list
        return valid_routes  # Return the list of valid routes

    def change_edge_hierarchieparents(self):
        """
        Updates the carrier type of certain edges in the graph to reflect hierarchical relationships.

        Behavior:
            - Iterates over all vertices in the graph.
            - For each vertex, checks if it is connected to any edge with carrier type "Product".
            - If such a connection exists:
                - Finds the edge where the vertex is the source and the carrier is "hasparent".
                - Changes the carrier of that edge to "hashierarchieparent".

        Purpose:
            - Reclassifies parent-child relationships for vertices involved in product-related connections.
            - Helps distinguish hierarchical structure from general parent links.
        """
        # Iterate over all vertices
        for vertex in self.g.vs:
            edges = self.g.es.select(_source=vertex.index)
            target_edges = self.g.es.select(_target=vertex.index)
            
            # If vertex is the source or target of an edge with carrier="Product"
            if any(edge['carrier'] == "Product" for edge in edges) or any(edge['carrier'] == "Product" for edge in target_edges):
                # Get the single edge with carrier="hasparent" that the vertex is the source of
                hasparent_edges = [edge for edge in edges if edge['carrier'] == "hasparent"]
                
                # If such edge exists, change its carrier to "hashierarchieparent"
                if len(hasparent_edges) > 0:
                    hasparent_edges[0]['carrier'] = "hashierarchieparent"

                                                                            

    def generate_graphfromAML(self, amlfilepath):
        """
        Generates a graph from an AutomationML (AML) file.
            amlfilepath (str): The file path to the AML file.
        Performs:
            - Processes XML elements and populates various mappings based on element tags and attributes.
            - Adds vertices and edges to the graph based on the mappings.
            - Establish relationships and hierarchies between the vertices
            - Applies  first order physical principle rules to the graph to find possible alarm propagation paths.
            - Identifies and processes control loops, cooling loops, and alarm paths.
            - Identifies valves and applies weights to their relationships in the graph.
        """
        tree = ET.parse(amlfilepath)
        root = tree.getroot()
        ie = {}
        internal_links = {}
        parent_map = {}
        attribute_map = {}
        elementsID = {}
        interface_map = {}
 
        def process_element(parent):
            """
            Recursively processes XML elements and populates various mappings based on element tags and attributes.
            Args:
                parent (xml.etree.ElementTree.Element): The parent XML element to process.
            Populates:
                elementsID (dict): Maps 'Name' attributes to 'ID' attributes for InternalElement tags.
                parent_map (dict): Maps child 'Name' attributes to parent 'Name' attributes for InternalElement tags.
                attribute_map (dict): Maps concatenated parent 'Name' and child 'Name' attributes to True for specific Attribute tags.
                interface_map (dict): Maps parent 'Name' attributes to dictionaries of ExternalInterface 'ID' and 'Name' attributes.
            Handles:
                InternalElement tags: Maps 'Name' to 'ID' and parent-child relationships.
                Attribute tags: Maps specific attributes related to measurements and states.
                ExternalInterface tags: Maps interface IDs and names to parent names.
            Raises:
                Exception: Catches and prints any exceptions that occur during processing.
            """

            for child in parent:
                try:
                    if child.tag == "{http://www.dke.de/CAEX}InternalElement":
                        elementsID[child.attrib['Name']] = child.attrib['ID']
                        if parent.attrib['Name'] != child.attrib['Name']:
                            parent_map[child.attrib['Name']] = parent.attrib['Name']
                        process_element(child)
                            
                    if child.tag == "{http://www.dke.de/CAEX}Attribute":
                        if child.attrib['Name'] in ["Temperature", "Pressure", "Flow", "Level", "Power", "Measurement_Power", "Measurement_Temperature", "Measurement_Pressure", "Measurement_Flow", "Measurement_Level", "State", "Measurement_State"] :#Concentration is removed because it will written separately
                            temp_str = parent.attrib['Name'] + '_' + child.attrib['Name']
                            attribute_map[temp_str] = True
                        # Check if the attribute name includes "Concentration" or "Material".
                        # This is used to trace nested attributes where "Concentration" leads to a Material type.
                        elif any(symbol in child.attrib['Name'] for symbol in ["Concentration", "Material"]):

                            for concentration_child in child:
                                if concentration_child.tag == "{http://www.dke.de/CAEX}Attribute":
                                    concentration_child_name=concentration_child.attrib['Name']
                                    temp_str_concentration=f"{parent.attrib['Name']}_{concentration_child_name}"
                                    attribute_map[temp_str_concentration]=True
                    if child.tag == "{http://www.dke.de/CAEX}ExternalInterface":
                        # Code to process ExternalInterface elements
                        # populate interface_map as needed
                        if parent.attrib['Name'] not in interface_map:
                            interface_map[parent.attrib['Name']] = {} 
                        interface_map[parent.attrib['Name']][child.attrib['ID']] = child.attrib['Name']

                except Exception as e:
                    print(f"Error processing element: {e}") 


        for elem in root.iter('{http://www.dke.de/CAEX}InternalElement'):
            process_element(elem)

        
        for key in parent_map:
           
            self.add_myvertex(key, 'hierarch', elementsID[key], False, False, False)
            if not key=='Plant': 
                if parent_map[key] in elementsID:
                    if len(self.g.vs.select(Tag=parent_map[key]))==0:
                    
                        self.add_myvertex(parent_map[key], 'hierarch', elementsID[parent_map[key]], False, False, False)
                    self.g.add_edge(self.g.vs["Tag"].index(key), self.g.vs["Tag"].index(parent_map[key]), carrier='hasparent', weight=0)
                    self.g.add_edge(self.g.vs["Tag"].index(parent_map[key]), self.g.vs["Tag"].index(key), carrier='haschild', weight=0)
                    
        # Extract base keys from attribute names by removing measurement-related suffixes.
        # "Concentration" attributes are handled separately to preserve their base name.
        remove_list = ["_Temperature", "_Pressure", "_Flow", "_Level", "_Power", "_Measurement_Power",  "_Measurement_Temperature", "_Measurement_Pressure", "_Measurement_Flow", "_Measurement_Level", "_Concentration", "_Measurement_Concentration", "_State", "_Measurement_State"]
        pattern = '|'.join(remove_list)
        for attr in attribute_map:
            if "Concentration" in attr:
                key=attr.split('_')[0]
            else:
                key = re.sub(pattern, '', attr)
            
            if key in elementsID:                        
                carrierstring = attr.replace(key+'_','')
                self.add_myvertex(attr, carrierstring, None, False, False, False)
                self.g.add_edge(self.g.vs["Tag"].index(attr), self.g.vs["Tag"].index(key), carrier='hasparent', weight=0)
                self.g.add_edge(self.g.vs["Tag"].index(key), self.g.vs["Tag"].index(attr), carrier='haschild', weight=0)

        def find_concentration_sensors(attribute_map):
            """
            Identifies and returns a list of unique sensor names related to "Concentration" 
            from the provided attribute map.

            This function scans through the attribute map (a list of strings) and collects 
            all entries that contain the keyword "Concentration", which typically indicates 
            a concentration sensor. If other sensor types are relevant, the logic can be 
            extended to include additional keywords.

            Parameters:
            - attribute_map (List[str]): A list of attribute names, typically strings 
            representing sensor identifiers.

            Returns:
            - List[str]: A list of unique sensor names that include the keyword "Concentration".
            """
            concentration_sonsors = []

            for attr in attribute_map:
                if "Concentration" in attr:
                    sensor_name = attr  # Extracts the base pattern
                    concentration_sonsors.append(sensor_name)

            return list(set(concentration_sonsors))  # Returns unique patterns
        #concentration_measurements are all Concentration Sensors
        concentration_measurement=find_concentration_sensors(attribute_map)
        

        def get_sensor_tag(sensor_name):
            """
            Extracts the sensor tag from a standardized sensor attribute string.

            Parameters:
                sensor_name (str): A string formatted as "<Tag>_<MeasurementType>_<Unit>", 
                                e.g., "AIR100_Concentration_C".

            Returns:
                str: The sensor tag portion of the string, e.g., "AIR100".

            Example:
                get_sensor_tag("AIR100_Concentration_C") → "AIR100"
            """
            sensor_tag=sensor_name.split("_")[0]
            return sensor_tag
        
        def get_concentration_type(sensor_name):
            """
            Extracts the concentration type from a standardized sensor attribute string.

            Parameters:
                sensor_name (str): A string formatted as "<Tag>_<MeasurementType>_<Unit>", 
                                e.g., "AIR100_Concentration_C".

            Returns:
                str: The concentration type portion of the string, e.g., "Concentration_C".

            Example:
                get_concentration_type("AIR100_Concentration_C") → "Concentration_C"
            """
            concentration_type="_".join(sensor_name.split("_")[1:])
            return concentration_type

        def add_concentration_itself_edges(self, concentration_sensors):
            """
            Adds directed edges between concentration-related vertices that share the same sensor tag
            but represent different concentration types.

            Parameters:
                concentration_sensors (List[str]): A list of sensor attribute strings, each formatted as
                                                "<Tag>_<ConcentrationType>", e.g., "AIR100_Concentration_A".

            Behavior:
                - For each pair of sensors:
                    - If they share the same sensor tag (e.g., "AIR100") but have different concentration types,
                    an edge is added from one to the other.
                - The edge is labeled with:
                    - carrier = "Concentration"
                    - lambda_factor = -1
                    - weight = 0.9

            Example:
                Adds an edge from "AIR100_Concentration_B" to "AIR100_Concentration_A"
                if both exist in the graph and have different concentration types.
            """
            for i in range(len(concentration_sensors)):
                for j in range(len(concentration_sensors)):
                    if i!=j:    
                        source_tag = concentration_sensors[i]
                        target_tag= concentration_sensors[j]
                        if get_sensor_tag(source_tag)==get_sensor_tag(target_tag):
                            if get_concentration_type(source_tag)!=get_concentration_type(target_tag):
                                source_index = self.g.vs.find(Tag=source_tag).index
                                target_index = self.g.vs.find(Tag=target_tag).index
                                self.g.add_edge(source=source_index, target=target_index, carrier="Concentration", lambda_factor=-1, weight=0.9)

        add_concentration_itself_edges(self,concentration_measurement)

        def find_myparent(child_name):
            """
            Searches the CAEX XML structure to find the parent InternalElement of a child element
            with the specified name.

            Parameters:
                child_name (str): The name of the child InternalElement to search for.

            Returns:
                str or None: The name of the parent InternalElement if found; otherwise, None.

            Behavior:
                - Iterates through all InternalElement tags in the XML.
                - For each parent element, checks its children to see if any match the given name.
                - If a match is found, returns the parent's "Name" attribute.
            """
            for parent in root.iter('{http://www.dke.de/CAEX}InternalElement'):
                for child in parent:
                    if child.tag.startswith('{http://www.dke.de/CAEX}') and "Name" in child.attrib and child.attrib["Name"] == child_name:
                        return parent.attrib.get("Name")  # Return the parent's name attribute
            return None  # Return None if no parent is found
        
        def extract_partners(root, interface_map):
            """
            Extracts partner connections from a CAEX XML structure by identifying InternalLinks
            between ExternalInterfaces that represent "Product" relationships.

            Parameters:
                root (Element): The root element of the parsed CAEX XML tree.
                interface_map (dict): A dictionary mapping InternalElement names to lists of ExternalInterface IDs.

            Returns:
                Tuple[List[str], List[str]]:
                    - partnerA_list: Names of InternalElements connected via RefPartnerSideA.
                    - partnerB_list: Names of InternalElements connected via RefPartnerSideB.

            Behavior:
                - Scans all ExternalInterface elements and collects those whose "Name" contains "Product".
                - Iterates over InternalLink elements and checks if both RefPartnerSideA and RefPartnerSideB
                refer to valid "Product" interfaces.
                - Uses the interface_map to resolve which InternalElement each interface ID belongs to.
                - Appends the resolved names to partnerA_list and partnerB_list respectively.

            Example:
                If "AIR100" and "V167" are connected via "Product" interfaces, they will be added to the lists.
            """
            partnerA_list = []
            partnerB_list = []

            # Step 1: Create a lookup set of ExternalInterface IDs that have "Product" in their name
            product_interface_ids = {elem.attrib["ID"] for elem in root.iter('{http://www.dke.de/CAEX}ExternalInterface') if "Product" in elem.attrib.get("Name", "")}

            # Step 2: Filter InternalLinks based on valid 'Product' connections
            for connect in root.iter('{http://www.dke.de/CAEX}InternalLink'):
                refA = connect.attrib.get("RefPartnerSideA")
                refB = connect.attrib.get("RefPartnerSideB")

                # Only process connections if BOTH partners are valid 'Product' interfaces
                if refA in product_interface_ids and refB in product_interface_ids:
                    name_partnerA = next((name for name, interfaces in interface_map.items() if refA in interfaces), None)
                    name_partnerB = next((name for name, interfaces in interface_map.items() if refB in interfaces), None)

                    if name_partnerA and name_partnerB:
                        partnerA_list.append(name_partnerA)
                        partnerB_list.append(name_partnerB)

            return partnerA_list, partnerB_list

        partnerA_list, partnerB_list = extract_partners(root, interface_map)
        
        from collections import defaultdict, deque

        def find_elements_between(partnerA_list, partnerB_list, start_partner, end_partner):
            """
            Finds all unique elements that lie on any path between two specified partners
            in a directed graph constructed from partner connections.

            Parameters:
                partnerA_list (List[str]): List of source elements (RefPartnerSideA).
                partnerB_list (List[str]): List of target elements (RefPartnerSideB).
                start_partner (str): The starting element in the graph.
                end_partner (str): The ending element in the graph.

            Returns:
                List[str]: A flattened list of unique elements that appear on any path
                        from start_partner to end_partner, in traversal order.

            Behavior:
                - Constructs a directed graph from the partner lists.
                - Uses breadth-first search (BFS) to find all paths from start to end.
                - Flattens all paths into a single list of unique elements.

            Example:
                If partnerA_list = ["A", "B"], partnerB_list = ["B", "C"],
                and start_partner = "A", end_partner = "C",
                the result will be ["A", "B", "C"].
            """
            # Step 1: Build the graph
            graph = defaultdict(list)
            for a, b in zip(partnerA_list, partnerB_list):
                graph[a].append(b)  # Directed edge from a -> b

            # Step 2: Find all paths using BFS
            def bfs(graph, start_partner, end_partner):
                queue = deque([(start_partner, [])])  # (current_node, path_so_far)
                paths = []  # Store all valid paths
                visited = set()  # Prevent revisiting nodes in the current path

                while queue:
                    current_partner, path = queue.popleft()

                    # Mark the current node as visited for the current path
                    visited.add(current_partner)

                    # Check all neighbors (connected nodes)
                    for neighbor in graph[current_partner]:
                        new_path = path + [(current_partner, neighbor)]  # Extend the path

                        # If we reach the end_partner, save this path
                        if neighbor == end_partner:
                            paths.append(new_path)
                        elif neighbor not in visited:
                            queue.append((neighbor, new_path))

                return paths

            paths = bfs(graph, start_partner, end_partner)

            # Step 3: Flatten the nested list of pairs
            def flatten_path(nested_pairs):
                flattened = []
                seen = set()  # Track visited elements

                for path in nested_pairs:
                    for start, end in path:
                        if start not in seen:
                            flattened.append(start)
                            seen.add(start)
                        if end not in seen:
                            flattened.append(end)
                            seen.add(end)

                return flattened


            return flatten_path(paths)
                

        def get_ref_base_system_unit_path(element_name):
            """
            Retrieves the class type of an InternalElement from an AML (AutomationML) structure,
            based on its 'RefBaseSystemUnitPath' attribute.

            Parameters:
                element_name (str): The name of the InternalElement whose class is to be retrieved.

            Returns:
                str or None: The value of the 'RefBaseSystemUnitPath' attribute, which indicates the
                            element's class or type in the AML hierarchy. Returns None if not found.

            Behavior:
                - Iterates through all <InternalElement> tags in the CAEX XML.
                - Matches the 'Name' attribute to the provided element name.
                - If a match is found, returns the associated 'RefBaseSystemUnitPath' class.

            Example:
                If an element named "V167" has RefBaseSystemUnitPath="Valve", the function returns "Valve".
            """
            for elem in root.iter('{http://www.dke.de/CAEX}InternalElement'):
                if elem.attrib['Name'] == element_name:
                    return elem.attrib.get('RefBaseSystemUnitPath', None)
            return None
        

        def get_value(element_name, attribute_name_partial):
            """
            Retrieves the value of a sensor attribute from a specified InternalElement in a CAEX XML structure.

            Parameters:
                element_name (str): The name of the InternalElement to search for.
                attribute_name_partial (str): A partial or formatted name of the attribute to match.
                                            Underscores and case differences are ignored during comparison.

            Returns:
                str or None: The text content of the matching <Value> element, or None if not found.

            Behavior:
                - Iterates through InternalElements to find one with the given name.
                - Searches its <Attribute> elements for a name that matches the given partial name,
                ignoring underscores and case.
                - If a match is found, returns the text content of its <Value> child element.

            Notes:
                - Matching is done by normalizing both attribute names (removing underscores and converting to lowercase).
                - This compensates for inconsistencies in AML formatting.

            Example:
                get_value("AIR100", "Measurement_Concentration") → "22.5"
            """
            namespace = "{http://www.dke.de/CAEX}"  # Define the namespace
            for elem in root.iter(f'{namespace}InternalElement'):
                if elem.attrib['Name'] == element_name:
                    # Search for an Attribute with a Name
                    for attribute in elem.iter(f'{namespace}Attribute'):
                        attribute_name = attribute.attrib.get('Name', "")
                        # Changing the name form. normally it should be diectly same form aml but i will chack why we loose "_"
                        if attribute_name.replace("_", "").lower() == attribute_name_partial.replace("_", "").lower():
                            # Find the Value and return it
                            value_elem = attribute.find(f'{namespace}Value')
                            if value_elem is not None:
                                return value_elem.text
            return None


        def edgestate_vertexes (concentration_sensors):
            """
            Returns a list of EdgeState vertex names.
            Checks for valid paths between pairs of concentration sensors with a valve in between.
            Adds an edge if both sensors have the same concentration type, different tags, 
            and their values are greater than 0.

            Args:
                concentration_sensors (List[str]): List of concentration sensor identifiers.

            Returns:
                List[str]: List of EdgeState vertex names in the format 
                        "Edge_State_<SensorA>--<SensorB>".
            """
            edge_state_vertexes_list=[]
            for i in range (len(concentration_sensors)):
                sensor_first_temp=concentration_sensors[i]
                sensor_first_tag=get_sensor_tag(sensor_first_temp)
                sensor_first_concentration=get_concentration_type(sensor_first_temp)
                source_sensor_value=get_value(sensor_first_tag,sensor_first_concentration)
                for j in range(len(concentration_sensors)):
                    if i!=j:   
                        sensor_second_temp=concentration_sensors[j]
                        sensor_second_tag=get_sensor_tag(sensor_second_temp)
                        sensor_second_concentration=get_concentration_type(sensor_second_temp)
                        target_sensor_value=get_value(sensor_second_tag,sensor_second_concentration)
                        if sensor_first_tag!=sensor_second_tag and sensor_first_concentration==sensor_second_concentration:
                            if sensor_first_tag not in partnerA_list:
                                sensor_first=find_myparent(sensor_first_tag)
                            else:
                                sensor_first=sensor_first_tag
                            if sensor_second_tag not in partnerB_list:
                                sensor_second=find_myparent(sensor_second_tag)
                            else:
                                
                                sensor_second=sensor_second_tag
                            
                            result=find_elements_between(partnerA_list,partnerB_list,sensor_first,sensor_second)
                            
                            #Adding edges between Concentration Sensors if Value is bigger than 0(AIR100_Concentration_A--AIR003_Concentration_A)
                            if len(result)!=0:
                                if (float(source_sensor_value)!=0):
                                    if (float(target_sensor_value)!=0):
                                        source_tag=sensor_first_temp
                                        target_tag =sensor_second_temp
                                        new_tag=(f"Edge_State_{source_tag}--{target_tag}")
                                        edge_state_vertexes_list.append(new_tag)
            return edge_state_vertexes_list


        edge_state_list=edgestate_vertexes(concentration_measurement)
        #Adding Dummy Vertexes Between Concentration Sensors
        for i in range(len(edge_state_list)):
            aml_id = f"10A{i}"  # Generate unique AMLID for each vertex
            self.add_myvertex(edge_state_list[i], in_carriertype=get_concentration_type(edge_state_list[i].split("--")[0]), in_amlID=aml_id,
                      in_alarmlimits=False, in_alarmstate=False, in_alarmtag=False,)


        def connection_between_concentration_sensors(self, concentration_sensors, partnerA_list, partnerB_list):
            """
            Constructs graph edges between concentration sensors based on shared concentration types,
            physical connectivity, and sensor values. Ensures deterministic behavior by sorting inputs
            and applying consistent selection logic.
            """
            # Sort concentration sensors to ensure consistent pair ordering
            concentration_sensors = sorted(concentration_sensors)

            for i in range(len(concentration_sensors)):
                sensor_first_temp = concentration_sensors[i]
                sensor_first_tag = get_sensor_tag(sensor_first_temp)
                sensor_first_concentration = get_concentration_type(sensor_first_temp)
                source_sensor_value = get_value(sensor_first_tag, sensor_first_concentration)

                for j in range(len(concentration_sensors)):
                    if i == j:
                        continue

                    sensor_second_temp = concentration_sensors[j]
                    sensor_second_tag = get_sensor_tag(sensor_second_temp)
                    sensor_second_concentration = get_concentration_type(sensor_second_temp)
                    target_sensor_value = get_value(sensor_second_tag, sensor_second_concentration)

                    # Check concentration type match and distinct sensors
                    if sensor_first_tag != sensor_second_tag and sensor_first_concentration == sensor_second_concentration:
                        sensor_first = sensor_first_tag if sensor_first_tag in partnerA_list else find_myparent(sensor_first_tag)
                        sensor_second = sensor_second_tag if sensor_second_tag in partnerB_list else find_myparent(sensor_second_tag)

                        result = find_elements_between(partnerA_list, partnerB_list, sensor_first, sensor_second)

                        # Sort result for deterministic valve selection
                        result = sorted(result)
                        between_valve_list = []
                        for idx in range(len(result) - 1):
                            ref_base_unit = get_ref_base_system_unit_path(result[idx])
                            if "Valve" in ref_base_unit:
                                between_valve_list.append(result[idx])
                                neighbor_valve = result[idx + 1]  # Deterministic: always use next in sorted list

                        # Add concentration edges if both sensors have non-zero values
                        if result and float(source_sensor_value) != 0 and float(target_sensor_value) != 0:
                            source_tag = sensor_first_temp
                            target_tag = sensor_second_temp
                            new_tag = f"Edge_State_{source_tag}--{target_tag}"

                            source_index = self.g.vs.find(Tag=source_tag).index
                            target_index = self.g.vs.find(Tag=target_tag).index
                            new_index = self.g.vs.find(Tag=new_tag).index

                            self.g.add_edge(source=source_index, target=new_index,
                                            carrier=sensor_first_concentration, lambda_factor=1, weight=1)
                            self.g.add_edge(source=new_index, target=target_index,
                                            carrier=sensor_first_concentration, lambda_factor=1, weight=1)

                        # Add flow and valve-state edges if valves are present
                        if between_valve_list and float(source_sensor_value) != 0 and float(target_sensor_value) != 0:
                            source_tag = sensor_first_temp
                            target_tag = sensor_second_temp
                            middle_index = self.g.vs.find(Tag=neighbor_valve + "_Flow").index
                            target_index = self.g.vs.find(Tag=target_tag).index

                            self.g.add_edge(source=middle_index, target=target_index,
                                            carrier=sensor_first_concentration, lambda_factor=1, weight=0.9)

                            edge_state_tag = f"Edge_State_{source_tag}--{target_tag}"
                            edge_state_index = self.g.vs.find(Tag=edge_state_tag).index

                            for valve_tag in sorted(between_valve_list):
                                valve_index = self.g.vs.find(Tag=valve_tag + "_State").index
                                self.g.add_edge(source=valve_index, target=edge_state_index,
                                                carrier="Undecided", lambda_factor=0, weight=1)

                    

        connection_between_concentration_sensors(self,concentration_measurement,partnerA_list,partnerB_list)

        def find_thermal_products(root):
            """
            Identifies InternalElements in a CAEX XML structure that are involved in thermal exchange.

            Parameters:
                root (Element): The root element of the CAEX XML tree.

            Returns:
                List[str]: A list of InternalElement names that are associated with thermal contact.

            Criteria for Inclusion:
                - The InternalElement contains an <Attribute> with Name="Temperature".
                - At least one of its <ExternalInterface> elements has:
                    - "Thermal" in its "Name" attribute, or
                    - "Thermal" in its "RefBaseClassPath" attribute.

            Behavior:
                - Iterates through all <InternalElement> nodes.
                - Checks for the presence of a temperature attribute.
                - If found, inspects associated external interfaces for thermal relevance.
                - Adds qualifying elements to the result list.

            Example:
                If an element has a temperature attribute and an interface named "ThermalContact",
                it will be included in the output list.
            """
            thermal_product_list = []
            namespace = "{http://www.dke.de/CAEX}"
            for elem in root.iter(f'{namespace}InternalElement'):
                # Check if it has an Attribute with Name="Temperature"
                has_temperature = any(
                    attr.attrib.get("Name") == "Temperature"
                    for attr in elem.findall(f'{namespace}Attribute')
                )

                if has_temperature:
                    for ext_interface in elem.findall(f'{namespace}ExternalInterface'):
                        name = ext_interface.attrib.get("Name", "")
                        path = ext_interface.attrib.get("RefBaseClassPath", "")
                        if "Thermal" in name or "Thermal" in path:
                            thermal_product_list.append(elem.attrib.get("Name", "Unknown"))
                            break  # Found one, no need to check more interfaces

            return thermal_product_list
        
        thermal_product_list=find_thermal_products(root)
   
        
        def find_all_neighbors(
            element_name, partnerA_list, partnerB_list, root, namespace="{http://www.dke.de/CAEX}"
        ):
            """
            Identifies neighboring InternalElements connected to a specified element in a CAEX XML structure.

            Parameters:
                element_name (str): The name of the target InternalElement.
                partnerA_list (List[str]): List of RefPartnerSideA values from InternalLinks.
                partnerB_list (List[str]): List of RefPartnerSideB values from InternalLinks.
                root (Element): The root element of the CAEX XML tree.
                namespace (str): The XML namespace used for CAEX elements (default is CAEX standard).

            Returns:
                Dict[str, List[str]]: A dictionary with two keys:
                    - "previous": List of InternalElements that link to the target element (incoming edges).
                    - "next": List of InternalElements that the target element links to (outgoing edges).

            Behavior:
                - Constructs directed adjacency maps from partnerA and partnerB lists.
                - Filters neighbors to include only valid InternalElement names found in the XML.
                - Returns neighbors grouped by directionality.

            Example:
                If "V167" connects to "AIR100", the result will include:
                    {
                        "previous": [...],
                        "next": ["AIR100"]
                    }
            """
            from collections import defaultdict

            # Step 1: Build adjacency maps
            outgoing = defaultdict(list)
            incoming = defaultdict(list)

            for a, b in zip(partnerA_list, partnerB_list):
                outgoing[a].append(b)
                incoming[b].append(a)

            # Step 2: Lookup all InternalElements by name
            element_names = {
                elem.attrib.get("Name") for elem in root.iter(f"{namespace}InternalElement")
            }

            # Step 3: Collect neighbors by direction
            neighbors = {
                "previous": [],
                "next": []
            }

            for direction, mapping in [("previous", incoming), ("next", outgoing)]:
                for neighbor_name in mapping.get(element_name, []):
                    if neighbor_name in element_names:
                        neighbors[direction].append(neighbor_name)

            return neighbors

        def find_sensors_in_element(vertex_name, root, sensor_type, NS="{http://www.dke.de/CAEX}"):
            """
            Identifies sensor-related attributes within a specified InternalElement in a CAEX XML structure.

            Parameters:
                vertex_name (str): The name of the parent InternalElement to search within.
                root (Element): The root of the CAEX XML tree, parsed using ElementTree.
                sensor_type (str): A keyword used to filter sensor attributes (e.g., "Temperature", "Pressure").
                NS (str): The XML namespace used in the CAEX document. Defaults to "{http://www.dke.de/CAEX}".

            Returns:
                List[str]: A list of strings formatted as "ChildElementName_AttributeName" for each matching
                        sensor attribute found within nested InternalElements.

            Behavior:
                - Locates the InternalElement with the specified name.
                - Searches its nested InternalElements for <Attribute> tags.
                - Filters attributes whose names contain both "Measurement" and the specified sensor_type.
                - Constructs a list of identifiers combining the child element name and attribute name.

            Example:
                If an InternalElement named "AIR100" contains a child with an attribute "Concentration_C",
                the result will include "AIR100_Concentration_C".
            """
            results = []

            # Find the InternalElement that matches the given vertex_name
            for internal_element in root.findall(f".//{NS}InternalElement"):
                if internal_element.get("Name") != vertex_name:
                    continue

                # Search for nested InternalElements within the target element
                for child_element in internal_element.findall(f".//{NS}InternalElement"):
                    child_name = child_element.get("Name")

                    # Look for Attributes inside the child InternalElement
                    for attribute in child_element.findall(f"{NS}Attribute"):
                        attr_name = attribute.get("Name")

                        # Match both "Measurement" and the sensor_type keyword
                        if "Measurement" in attr_name and sensor_type in attr_name:
                            results.append(f"{child_name}_{attr_name}")

            return results


        def find_coolest_sensor_element(base_element):
            """
            Identifies the neighboring InternalElement with the lowest temperature value relative to a given base element.

            Parameters:
                base_element (str): The name of the InternalElement from which the search begins.

            Returns:
                str or None: The name of the coolest neighboring element, or None if:
                    - The base element has the lowest temperature.
                    - All temperature values are equal.
                    - No temperature data is available.

            Behavior:
                - Searches for temperature sensor values in:
                    - The base element itself.
                    - Its first-level neighbors (previous and next).
                    - Its second-level neighbors (neighbors of neighbors).
                - Uses `find_sensors_in_element` to locate temperature sensors.
                - Uses `get_value` to retrieve temperature readings.
                - Uses `find_all_neighbors` to traverse the graph structure.
                - Compares all collected temperature values and returns the name of the coolest neighbor.

            Notes:
                - If the coolest element is a second-level neighbor, the function returns the first-level link element
                that connects to it.
                - Temperature values are assumed to be numeric and comparable.

            Example:
                If "Condenser_Product_Side_Temperature" has a temperature of 40°C and its neighbor "Condenser_Water_Side_Temperature" has 25°C,
                the function will return "Condenser_Water_Side_Temperature".
            """
            checked_elements = []
            sensor_values = {}

            
            def try_add_sensor_value(element, label):
                sensors = find_sensors_in_element(element, root, "Temperature")
                if sensors:
                    value = get_value(element, "Measurement_Temperature")
                    if value is not None:
                        sensor_values[label] = float(value)
                        checked_elements.append((label, element))

            # 1 Check self
            try_add_sensor_value(base_element, "self")

            # 2 Previous elements (1 and 2 steps)
            prev_elements = find_all_neighbors(base_element, partnerA_list, partnerB_list, root, namespace="{http://www.dke.de/CAEX}")["previous"]
            for i, prev in enumerate(prev_elements):
                try_add_sensor_value(prev, f"prev_{i}")
                prev_prev = find_all_neighbors(prev, partnerA_list, partnerB_list, root, namespace="{http://www.dke.de/CAEX}")["previous"]
                for j, pp in enumerate(prev_prev):
                    try_add_sensor_value(pp, f"prev_prev_{i}_{j}")

            # 3 Next elements (1 and 2 steps)
            next_elements = find_all_neighbors(base_element, partnerA_list, partnerB_list, root, namespace="{http://www.dke.de/CAEX}")["next"]
            for i, nxt in enumerate(next_elements):
                try_add_sensor_value(nxt, f"next_{i}")
                nxt_nxt = find_all_neighbors(nxt, partnerA_list, partnerB_list, root, namespace="{http://www.dke.de/CAEX}")["next"]
                for j, nn in enumerate(nxt_nxt):
                    try_add_sensor_value(nn, f"next_next_{i}_{j}")

            if not sensor_values:
                return None

            # Find the label of the minimum temperature
            lowest_label = min(sensor_values, key=sensor_values.get)

            # If it's the base element, return None
            if lowest_label == "self" :
                return None
            if (len(sensor_values)>1) and (len(set(sensor_values.values()))==1):
                return None
            
            # Otherwise return the first element one step away from base
            for label, el in checked_elements:
                if label == lowest_label:
                    # For second-level neighbors, return first link element
                    if label.startswith("prev_prev_"):
                        index = label.split("_")[2]
                        return prev_elements[int(index)]
                    if label.startswith("next_next_"):
                        index = label.split("_")[2]
                        return next_elements[int(index)]
                    return el  # Direct 1-step neighbor

            return None

        # Identify cooling actors for each thermal product by finding their coolest neighbor

        cooling_actor=[]
        for j in range(len(thermal_product_list)):
            seaching_element=thermal_product_list[j]
            coolest_actor=find_coolest_sensor_element(seaching_element)
            if coolest_actor is not None:
                cooling_actor.append(coolest_actor)

        # Create edges between connected elements based on interface direction (OUT, INOUT); log unhandled cases
        
        for connect in root.iter('{http://www.dke.de/CAEX}InternalLink'):
            name_partnerA = next((name for name, interfaces in interface_map.items() if connect.attrib['RefPartnerSideA'] in interfaces), None)
            name_partnerB = next((name for name, interfaces in interface_map.items() if connect.attrib['RefPartnerSideB'] in interfaces), None)
            if name_partnerA is not None and name_partnerB is not None:
                connect_partnerA = interface_map[name_partnerA][connect.attrib['RefPartnerSideA']]
                connect_partnerB = interface_map[name_partnerB][connect.attrib['RefPartnerSideB']]
                if '_OUT' in connect_partnerA and not 'OUT' in connect_partnerB:
                    self.g.add_edge(self.g.vs["Tag"].index(name_partnerA), self.g.vs["Tag"].index(name_partnerB), carrier=connect_partnerA.replace("_OUT",''), weight=0)
                else:
                    if 'INOUT' in connect_partnerA:
                        if 'INOUT' in connect_partnerB:
                            self.g.add_edge(self.g.vs["Tag"].index(name_partnerA), self.g.vs["Tag"].index(name_partnerB), carrier=connect_partnerA.replace("_INOUT",''), weight=0)
                            self.g.add_edge(self.g.vs["Tag"].index(name_partnerB), self.g.vs["Tag"].index(name_partnerA), carrier=connect_partnerA.replace("_INOUT",''), weight=0)
                        elif 'IN' in connect_partnerB:
                                print('ERROR: Case2 NOT IMPLEMENTED')
                        else:
                                print('ERROR: Case3 NOT IMPLEMENTED')
                    else:
                        if 'OUT' in connect_partnerB:
                            print('ERROR: Case4 NOT IMPLEMENTED')
        

        # Run hierarchical and functional path analysis; isolate cooling loops via thermal contact edges

        self.change_edge_hierarchieparents() 

        self.vertexpaths, loop1length = self.find_paths(['hasparent', 'Product', 'haschild'], [3,1,3])
        alarmpaths, loop2length = self.find_paths(['hasparent', 'haschild'], [3,3])
        cooling_loops = self.find_paths(['hasparent', 'Product_ThermalContact', 'haschild'], [5,1,5])[0:-1]
        cooling_loops=flatten(cooling_loops)[2:-1]
                           
        # Apply  first-order dynamic rules to model physical and actuator interactions across the graph
        
        # T(i-1) -> T(i) Product
        self.apply_rule("Temperature", "Temperature", strength=0.99, time_constant=0.0, effect_factor=1, findValves=False, rulenumber=1)
        # P(i-1) -> P(i) Product
        self.apply_rule("Pressure", "Pressure", strength=0.99, time_constant=0.0, effect_factor=1, findValves=False, rulenumber=2)
        # F(i-1) -> F(i) Product
        self.apply_rule("Flow", "Flow", strength=0.99, time_constant=0.0, effect_factor=1, findValves=False, rulenumber=3)
        # L(i-1) -> L(i) Product
        self.apply_rule("Level", "Level", strength=0.99, time_constant=0.0, effect_factor=1, findValves=False, rulenumber=4)
        # F(i-1) -> L(i) Product
        self.apply_rule("Flow", "Level", strength=0.99, time_constant=0.1, effect_factor=1, findValves=False, rulenumber=5)        
        # L(i-1) -> F(i) Product
        self.apply_rule("Level", "Flow", strength=0.99, time_constant=0.1, effect_factor=1, findValves=False, rulenumber=6)
        # P(i-1) -> T(i) Product
        self.apply_rule("Pressure", "Temperature", strength=0.1, effect_factor= -1, findValves=False, rulenumber=7)
        # V(i-1) -> F(i) Actuator
        self.apply_rule ("State", "Flow", strength=0.9, time_constant=0.0, effect_factor=1, findValves=False, rulenumber=8, apply_carrier='State')
        
        # T(i) -> T(i-1) Product
        self.apply_rule("Temperature", "Temperature", strength=0.99, time_constant=0.0, effect_factor=1, findValves=False, rulenumber=1, inverse=True)
        # P(i) -> P(i-1) Product
        self.apply_rule("Pressure", "Pressure", strength=0.99, time_constant=0.0, effect_factor=1, findValves=False, rulenumber=2, inverse=True)
        # F(i) -> F(i-1) Product
        self.apply_rule("Flow", "Flow", strength=0.99, time_constant=0.0, effect_factor=1, findValves=False, rulenumber=3, inverse=True)
        # L(i) -> L(i-1) Product
        self.apply_rule("Level", "Level", strength=0.99, time_constant=0.0, effect_factor=1, findValves=False, rulenumber=4, inverse=True)
        # F(i) -> L(i-1) Product
        self.apply_rule("Flow", "Level", strength=0.99, time_constant=0.1, effect_factor= -1, findValves=False, rulenumber=5, inverse=True)
        # L(i) -> F(i-1) Product
        self.apply_rule("Level", "Flow", strength=0.99, time_constant=0.1, effect_factor= -1, findValves=False, rulenumber=6, inverse=True)
        # T(i) -> P(i-1) Product
        #print(self.apply_rule("Temperature", "Pressure", strength=0.1, effect_factor= -1, findValves=False, rulenumber=7, inverse=True))#, parent_target='P1'))
        # V(i) -> F(i-1) Actuator
        self.apply_rule("Flow", "State", strength=0.9, time_constant=0.0, effect_factor=1, findValves=False, rulenumber=8, inverse=True, apply_carrier='State')
       

        # Apply alarm propagation rules from measurements to physical states; convert 'isalarm' edges to 'hasalarm'

        self.apply_rule("Measurement_Temperature", "Temperature", strength=1.00, time_constant=0.0, effect_factor=1, apply_carrier='isalarm', searchpath=alarmpaths, shortest=True, findValves=False, rulenumber=0)        
        self.apply_rule("Measurement_Pressure", "Pressure", strength=1.00, time_constant=0.0, effect_factor=1, apply_carrier='isalarm', searchpath=alarmpaths, shortest=True, findValves=False, rulenumber=0)
        self.apply_rule("Measurement_Flow", "Flow", strength=1.00, time_constant=0.0, effect_factor=1, apply_carrier='isalarm', searchpath=alarmpaths, shortest=True, findValves=False, rulenumber=0)
        self.apply_rule("Measurement_Level", "Level", strength=1.00, time_constant=0.0, effect_factor=1, apply_carrier='isalarm', searchpath=alarmpaths, shortest=True, findValves=False, rulenumber=0)
        self.apply_rule("Measurement_State", "State", strength=1.00, time_constant=0.0, effect_factor=1, apply_carrier='isalarm', searchpath=alarmpaths, shortest=True, findValves=False, rulenumber=0)
        self.apply_rule("Measurement_Power", "Power", strength=1.00, time_constant=0.0, effect_factor=1, apply_carrier='isalarm', searchpath=alarmpaths, shortest=True, findValves=False, rulenumber=0)
        [self.g.add_edge(edge.target, edge.source, carrier="hasalarm", weight=1, tau=0, lambda_factor = 1) for edge in self.g.es if edge['carrier'] == 'isalarm']

        # T(i-1) -> T(i) ThermaclContact
        self.apply_rule("Temperature", "Temperature", strength=0.90, time_constant=0.0, effect_factor=1, apply_carrier='thermal_connection', searchpath=cooling_loops, findValves=False, rulenumber=21)
        # F(i-1) -> T(i) ThermaclContact
        self.apply_rule("Flow", "Temperature", strength=0.85, time_constant=0.0, effect_factor= 1, apply_carrier='thermal_connection', searchpath=cooling_loops, findValves=False, rulenumber=22)
        
        # Identify cooling-related edges and invert their influence from Flow to Temperature by setting lambda_factor to -1

        # Step 1: Find all vertices whose Tag contains any cooling_actor substring
        cooling_actor_indices = [
            v.index for v in self.g.vs
            if any(actor_tag in v["Tag"] for actor_tag in cooling_actor)
        ]

        # Step 2: Select edges where the source is one of those vertices
        cooling_edges = self.g.es.select(_source_in=cooling_actor_indices)

        # Step 3: Change lambda factor to -1 
        for edge in cooling_edges:
            source_vertex = self.g.vs[edge.source]
            target_vertex = self.g.vs[edge.target]
            source_tag = source_vertex["Tag"]
            target_tag = target_vertex["Tag"]
            if source_vertex["Carrier"]=="Flow" and target_vertex["Carrier"]=="Temperature":
                edge["lambda_factor"]=-1

        # Define control loops and apply rules linking physical measurements to actuator states via digital pathways

        control_loops = self.find_paths(['hasalarm', 'hasparent', 'Digital', 'haschild'], [1,1,5,1])
        control_loops=flatten(control_loops)[2:-1]

        # Control Rules:
        # T(i-1) -> V(i) Digital
        self.apply_rule("Temperature", "State", strength=0.95, time_constant=0.0, effect_factor=0, apply_carrier='control', searchpath=control_loops, findValves=False, rulenumber=31)
        # P(i-1) -> V(i) Digital
        self.apply_rule("Pressure", "State", strength=0.95, time_constant=0.0, effect_factor=0, apply_carrier='control', searchpath=control_loops, findValves=False, rulenumber=32)
        # F(i-1) -> V(i) Digital
        self.apply_rule("Flow", "State", strength=0.95, time_constant=0.0, effect_factor= -1, apply_carrier='control', searchpath=control_loops, findValves=False, rulenumber=33)
        # L(i-1) -> V(i) Digital
        self.apply_rule("Level", "State", strength=0.95, time_constant=0.0, effect_factor=0, apply_carrier='control', searchpath=control_loops, findValves=False, rulenumber=34)

       # Instantiate Valve objects from graph vertices with 'State' carrier and apply their relational logic to the graph 
        
        self.valves = []
        for v in self.g.vs:
            if v['Carrier'] == 'State':
                temp_name = v['Tag'].replace("_State","")
                self.valves.append(Valve(temp_name))  
                print(self.valves[-1])      
        
        for valve in self.valves:
            valve.find_relations(self.g)
            valve.apply_weights(self.g)
            
        
        

##############################################################################################################################################################################################################################################################################
    def update_valves(self, in_valvedict):
        """
        Updates the state of valve objects and propagates their influence in the graph.

        Parameters:
            in_valvedict (Dict[str, Any]): A dictionary mapping valve tags to their new state values.

        Behavior:
            - Iterates through all Valve instances in self.valves.
            - If a valve's tag exists in the input dictionary:
                - Updates the valve's internal state using `update_state`.
                - Re-applies its influence on the graph using `apply_weights`.

        Example:
            If in_valvedict = {"VALVE_001": "open"}, the corresponding Valve object will update its state
            and adjust graph weights accordingly.
        """
        for valve in self.valves:
            if valve.Tag in in_valvedict:
                valve.update_state(in_valvedict[valve.Tag])
                valve.apply_weights(self.g)
    
    def evaluate_valvestate(self, in_valuedict):
        """
        Evaluates and updates the state of all valves using a provided value dictionary,
        and propagates their influence in the graph.

        Parameters:
            in_valuedict (Dict[str, Any]): A dictionary mapping each valve's tag to its corresponding state value.

        Behavior:
            - Iterates through all Valve instances in self.valves.
            - For each valve:
                - Updates its internal state using `update_state`.
                - Applies its influence on the graph using `apply_weights`.

        Assumptions:
            - All valve tags in self.valves are present in in_valuedict.
            - No missing keys or validation is performed.

        Example:
            If in_valuedict = {"VALVE_001": "closed", "VALVE_002": "open"},
            both valves will update their states and adjust graph weights accordingly.
        """
        for valve in self.valves:
            valve.update_state(in_valuedict[valve.Tag])
            valve.apply_weights(self.g)
    

    def modified_dijkstra(self, source_vertex_id, target_vertex_id, print_paths=True):
        """
        Computes a modified shortest path between two vertices in a directed graph using a variant of Dijkstra's algorithm.

        Modifications:
            - Separately tracks paths with positive and negative cumulative signs, based on vertex 'AS' attributes.
            - Transforms edge weights using -log(weight) to prioritize stronger connections.
            - Filters out paths with repeating patterns or mirrored segments to avoid loops.
            - Skips paths with low-weight edges, missing lambda factors, or excessive length.

        Parameters:
            source_vertex_id (int): Index of the source vertex in the graph.
            target_vertex_id (int): Index of the target vertex in the graph.
            print_paths (bool): Optional flag to enable path printing (currently unused).

        Returns:
            Tuple[float, List[int]]:
                - distance: The shortest path distance to the target vertex (based on transformed weights).
                - path: A list of vertex indices representing the selected path.

        Behavior:
            - Initializes separate distance and path trackers for positive and negative sign propagation.
            - Uses a priority queue to explore successors and update paths based on sign and weight.
            - Applies filtering to avoid invalid or cyclic paths.
            - Returns the shortest valid path that matches the sign of the target vertex's 'AS' attribute.

        Example:
            If vertex A has AS > 0 and vertex B has AS < 0, the algorithm will compute the shortest
            negatively signed path from A to B, avoiding loops and weak edges.
        """
        def has_duplicate_pattern(path):
            for i in range(len(path) - 3):
                if path[i:i+2] == path[i+2:i+4] or path[i] == path[i+2]:
                    return True
            return False
        
        positive_dist = [np.inf] * len(self.g.vs)
        negative_dist = [np.inf] * len(self.g.vs)
        positive_path = [[]] * len(self.g.vs)
        negative_path = [[]] * len(self.g.vs)

        if self.g.vs[source_vertex_id]['AS'] > 0:
            positive_dist[source_vertex_id] = 0
        elif self.g.vs[source_vertex_id]['AS'] < 0:
            negative_dist[source_vertex_id] = 0

        queue = [(0, source_vertex_id, self.g.vs[source_vertex_id]['AS'], [source_vertex_id])]

        while queue:
            (d, v, sign, path_so_far) = heapq.heappop(queue)

            for u in self.g.successors(v):
                edge_id = self.g.get_eid(v, u)
                edge_weight = self.g.es[edge_id]['weight']
                edge_lambda = self.g.es[edge_id]['lambda_factor']

                if edge_weight <= 0.1 or edge_lambda==None or len(path_so_far) > 10: #edge_weight <= 0.05 or edge_lambda==None or len(path_so_far) > 20:
                    continue
                                    
                edge_weight = -np.log(edge_weight)
                
                new_sign = sign * edge_lambda
                new_path = path_so_far + [u]

                # Inside your while loop, right after creating new_path
                if has_duplicate_pattern(new_path):
                    continue

                if new_sign >= 0 and d + edge_weight < positive_dist[u]:
                    positive_dist[u] = d + edge_weight
                    positive_path[u]   = new_path
                    heapq.heappush(queue, (positive_dist[u], u, 1, new_path))

                if new_sign <= 0 and d + edge_weight < negative_dist[u]:
                    negative_dist[u] = d + edge_weight
                    negative_path[u]   = new_path
                    heapq.heappush(queue, (negative_dist[u], u, -1, new_path))

        if self.g.vs[target_vertex_id]['AS'] > 0:
            distance = positive_dist[target_vertex_id]
            path = positive_path[target_vertex_id]
        elif self.g.vs[target_vertex_id]['AS'] < 0:
            distance = negative_dist[target_vertex_id]
            path = negative_path[target_vertex_id]
        else:
            return np.inf, []
        return distance



    def modified_dijkstra_WAI(self, source_vertex_id, target_vertex_id):
        """
        Computes a sign-aware shortest path distance from a source to a target vertex using a modified Dijkstra algorithm.

        Key Features:
            - Separately tracks distances for paths with positive and negative cumulative signs.
            - Edge weights are transformed using -log(weight) to prioritize stronger connections.
            - Path sign is propagated by multiplying the current sign with each edge's lambda_factor.
            - Only returns a valid distance if the sign of the path matches the alarm state ('AS') of the target vertex.

        Parameters:
            source_vertex_id (int): Index of the source vertex in the graph.
            target_vertex_id (int): Index of the target vertex in the graph.

        Returns:
            float: The shortest path distance to the target vertex with matching sign,
                or np.inf if no valid path exists.

        Behavior:
            - Initializes separate distance arrays for positive and negative sign propagation.
            - Uses a priority queue to explore successors and update distances.
            - Skips edges with zero weight and filters paths based on sign compatibility.
            - Returns the shortest distance that aligns with the target vertex's 'AS' attribute.

        Example:
            If the target vertex has AS = -1, only paths with negative cumulative sign will be considered.
        """
        # Initialize positive and negative distance lists
        positive_dist = [np.inf] * len(self.g.vs)
        negative_dist = [np.inf] * len(self.g.vs)

        # Start from source_vertex with zero distance in appropriate list
        if self.g.vs[source_vertex_id]['AS'] > 0:
            positive_dist[source_vertex_id] = 0
        elif self.g.vs[source_vertex_id]['AS'] < 0:
            negative_dist[source_vertex_id] = 0
        else:
            print("ERROR: Source Vertex has invalid alarmstate AS=0")

        queue = [(0, source_vertex_id, self.g.vs[source_vertex_id]['AS'])]

        while queue:
            (d, v, sign) = heapq.heappop(queue)
            
            for u in self.g.successors(v):
                edge_id = self.g.get_eid(v, u)
                edge_weight = self.g.es[edge_id]['weight']
                edge_lambda = self.g.es[edge_id]['lambda_factor']

                # Determine the sign of the product of the path sign and the lambda factor
                if edge_weight==0:
                    continue
                
                edge_weight=-np.log(edge_weight)
                new_sign = sign * edge_lambda
                # Check if product is positive or negative and update distance if new path is shorter
                # But only update the target node distance if the 'AS' value of the target matches the sign of the path                
                if new_sign >= 0 and d + edge_weight < positive_dist[u]:
                    positive_dist[u] = d + edge_weight
                    heapq.heappush(queue, (positive_dist[u], u, 1))
                if new_sign <= 0 and d + edge_weight < negative_dist[u]:
                    negative_dist[u] = d + edge_weight
                    heapq.heappush(queue, (negative_dist[u], u, -1))

        # return the shortest distance to the target_vertex_id or np.inf if it is not reachable
        if self.g.vs[target_vertex_id]['AS'] > 0:
            return positive_dist[target_vertex_id]
        elif self.g.vs[target_vertex_id]['AS'] < 0:
            return negative_dist[target_vertex_id]
        else:
            return np.inf


    def dfs(self, source_vertex_id, target_vertex_id, visited, weight):
        """
        Performs a depth-first search to find a weighted path from source to target in a directed graph.

        Parameters:
            source_vertex_id (int): Index of the starting vertex.
            target_vertex_id (int): Index of the target vertex.
            visited (List[bool]): A list tracking visited vertices to prevent cycles.
            weight (float): Initial cumulative weight of the path.

        Returns:
            Tuple[List[int], float]:
                - best_path: A list of vertex indices representing the most weighted path found.
                - max_weight: The cumulative weight of the best path.

        Behavior:
            - Recursively explores successors of the current vertex.
            - Multiplies edge weights along the path to compute cumulative strength.
            - Filters out edges with zero weight or excluded carriers (e.g., "hasparent", "Product").
            - Avoids revisiting nodes and prunes paths with low cumulative weight (< 0.1).
            - Returns the path with the highest cumulative weight from source to target.

        Notes:
            - Edge weights are not transformed (e.g., no logarithmic scaling).
            - This is a greedy DFS variant focused on maximizing path weight, not minimizing distance.

        Example:
            If vertex A connects to B and C, and A → C → B yields higher cumulative weight,
            the function will return that path.
        """
        visited[source_vertex_id] = True
        path = [source_vertex_id]
        if source_vertex_id == target_vertex_id:
            return path, weight
        else:
            max_weight = weight
            best_path = []
            exclude_carriers = ["hasparent", "haschild", "Product"]
            for v in self.g.successors(source_vertex_id):
                edge_id = self.g.get_eid(source_vertex_id, v)
                edge_weight = self.g.es[edge_id]['weight']
                edge_carrier = self.g.es[edge_id]['carrier']
                if edge_weight != 0 and edge_carrier not in exclude_carriers:
                    new_weight = weight * edge_weight
                    if not visited[v] and abs(new_weight) >= 0.1:
                        new_path, new_weight = self.dfs(v, target_vertex_id, visited, new_weight)
                        if new_weight > max_weight:
                            max_weight = new_weight
                            best_path = path + new_path
            visited[source_vertex_id] = False  # Mark the vertex as unvisited after visiting all its neighbors
            return best_path, max_weight


 

    def evaluate_alarm(self, source_vertex_id, target_vertex_id):
        """
        Evaluates the strongest causal path between two vertices in the graph using depth-first search (DFS).

        Parameters:
            source_vertex_id (int): Index of the source vertex in the graph.
            target_vertex_id (int): Index of the target vertex in the graph.

        Returns:
            Tuple[List[int], float]:
                - best_path: A list of vertex indices representing the strongest causal path.
                - max_weight: The cumulative weight of the best path.

        Behavior:
            - Initializes a visited list to track traversal and prevent cycles.
            - Invokes a modified DFS to explore all valid paths from source to target.
            - Filters edges based on carrier type and minimum weight threshold.
            - Accumulates edge weights multiplicatively to evaluate path strength.
            - Returns the path with the highest cumulative weight.

        Example:
            If multiple paths exist from a sensor to an actuator, the method returns the one with the strongest influence.
        """
        visited = [False] * self.g.vcount()
        best_path, max_weight = self.dfs(source_vertex_id, target_vertex_id, visited, 1.0)
        return best_path, max_weight

    
    def generate_alarmPLUT(self):
        """
        Generates a Pairwise Lookup Table (PLUT) for alarm propagation based on causal path weights.

        For each pair of vertices with non-zero alarm state ('AS'):
            - Computes the shortest causal path using a sign-aware Dijkstra algorithm.
            - Multiplies the path weight by the source and target vertices' 'AS' values.
            - Stores the resulting influence score in a 2D matrix.

        Returns:
            List[List[float]]: A square matrix where entry [i][j] represents the weighted influence
                            from vertex i to vertex j based on alarm propagation logic.

        Behavior:
            - Filters vertices with non-zero 'AS' values.
            - Initializes a zero-filled matrix of size NxN, where N is the number of alarm-active vertices.
            - Computes directional influence scores for each vertex pair (excluding self-pairs).
            - Uses `modified_dijkstra` to evaluate path strength and sign compatibility.

        Example:
            If vertex A has AS = +1 and vertex B has AS = -1, and the shortest path weight is 0.5,
            then PLUT[A][B] = +1 x 0.5 x -1 = -0.5.
        """
        # Get all vertices with 'AS' not equal to 0
        vertices = [v for v in self.g.vs if v["AS"] != 0]
        
        # Initialize the matrices
        weight_matrix = [[0]*len(vertices) for _ in range(len(vertices))]

        # Compute the paths and weights
        for i in range(len(vertices)):
            for j in range(len(vertices)):
                if i != j:
                    weight = self.modified_dijkstra(vertices[i].index, vertices[j].index)
                    weight_matrix[i][j] = vertices[i]["AS"] * weight * vertices[j]["AS"]

        return weight_matrix

    import numpy as np


    def find_paths(self, carriercondition, carriermax):
        """
        Identifies and filters directed paths in the graph based on carrier constraints and structural criteria.

        For each vertex in the graph:
            - Explores outgoing paths using `explore_edges`, constrained by carrier type and maximum occurrences.
            - Optionally checks for the presence of a specific carrier (e.g., "Product") using `check_path`.
            - Accumulates valid paths and computes their total length.
            - Filters out cyclic paths that start and end at the same vertex.

        Parameters:
            carriercondition (Union[str, List[str]]): Carrier type(s) used to guide edge exploration.
            carriermax (Union[int, List[int]]): Maximum allowed occurrences of each carrier type in a path.

        Returns:
            Tuple[List[List[int]], int]:
                - listofpaths: A list of valid paths (each as a list of vertex indices).
                - totallength: The sum of lengths of all valid paths.

        Notes:
            - The internal `check_path` function counts how many edges in a path match a specific carrier.
            - Paths with fewer than two matching "Product" edges are ignored by the checker but still included.

        Example:
            If carriercondition = ["hasparent", "Product", "haschild"] and carriermax = [3,1,3],
            the method returns all paths satisfying those constraints, excluding self-loops.
        """
        def check_path(g, path, carrier):
            vertex_list = []
            for j in path:
                vertex_list.append(g.vs(j)["Tag"])
            carrier_count = 0
            for i in range(1, len(path)):
                edge_id = g.get_eid(path[i-1], path[i])
                if g.es[edge_id]['carrier'] == carrier:
                    carrier_count += 1
            return carrier_count if carrier_count > 1 else 0
        
        totallenght = 0
        listofallpaths = []

        for i in range(len(self.g.vs)):  
            paths = self.explore_edges(i, carriercondition, carriermax)

            if paths:
                for path in paths:
                    checker = check_path(self.g, path, "Product")
                    if checker:
                        pass

                    totallenght += len(path)
                    listofallpaths.append(path)

        return [sublist for sublist in listofallpaths if sublist[0] != sublist[-1]], totallenght

    def explore_edges(self, source_vertex_id, carrierlist, carriermax, used_carriers=None, vertex_path=None, carrier_index=0, carrier_count=0):
        """
        Recursively explores directed paths in the graph based on ordered carrier constraints.

        Parameters:
            source_vertex_id (int): Index of the starting vertex.
            carrierlist (List[str]): Ordered list of carrier types to follow (e.g., ["hasparent", "Product", "haschild"]).
            carriermax (List[int]): Maximum allowed usage count for each carrier type in the path.
            used_carriers (List[str], optional): Carriers used so far (for recursive tracking).
            vertex_path (List[int], optional): Sequence of visited vertex indices (for recursive tracking).
            carrier_index (int): Index of the current carrier in carrierlist.
            carrier_count (int): Number of times the current carrier has been used.

        Returns:
            List[List[int]]: A list of valid paths (each as a list of vertex indices) that satisfy the carrier constraints.

        Behavior:
            - Starts from the source vertex and recursively traverses edges matching the current carrier.
            - Tracks how many times each carrier is used, enforcing the limits in carriermax.
            - Advances to the next carrier only after the current one has been used at least once.
            - Adds a path to the result once all carriers in carrierlist have been used.
            - Copies path and carrier state at each recursion step to avoid mutation.

        Notes:
            - This method is designed for structured path discovery in semantic graphs (e.g., CAEX-based models).
            - Paths are built incrementally and filtered based on carrier usage patterns.

        Example:
            If carrierlist = ["hasparent", "Product", "haschild"] and carriermax = [3,1,3],
            the method returns all paths that follow this sequence without exceeding the limits.
        """
        #Working Prefinal Verion DO NOT DELETE OR ALTER!
        all_paths = []
        if vertex_path is None:
            vertex_path = []
        if used_carriers is None:
            used_carriers = []

        vertex_path.append(source_vertex_id)

        if set(used_carriers) == set(carrierlist):
            all_paths.append(list(vertex_path))  # Copied the vertex_path to prevent mutation in recursive calls

        for edge in self.g.es.select(_source=source_vertex_id):
            if edge['carrier'] == carrierlist[carrier_index] and carrier_count < carriermax[carrier_index]:
                new_used_carriers = used_carriers.copy()  # Make a copy of used_carriers before appending
                new_used_carriers.append(edge['carrier'])  # Appended to the copy, not the original list
                # Passed copies of used_carriers and vertex_path to the recursive call
                new_paths = self.explore_edges(edge.target, carrierlist, carriermax, new_used_carriers, vertex_path.copy(), carrier_index, carrier_count + 1)
                if new_paths:
                    all_paths.extend(new_paths)
            # Updated condition to allow for the next carrier only if the current one has been used at least once
            if carrier_count >= 1 and carrier_index + 1 < len(carrierlist) and edge['carrier'] == carrierlist[carrier_index + 1]:
                # Passed copies of used_carriers and vertex_path to the recursive call                
                new_used_carriers = used_carriers.copy()  # Make a copy of used_carriers before appending
                new_used_carriers.append(edge['carrier'])  # Appended to the copy, not the original list
                new_paths = self.explore_edges(edge.target, carrierlist, carriermax, new_used_carriers, vertex_path.copy(), carrier_index + 1, 1) #Carriercount has to be 1, as this functioncall allready used the carrier once
                if new_paths:
                    all_paths.extend(new_paths)

        return all_paths

    def print_edge_vertices(self):
        """
        Prints a formatted summary of all edges in the graph, including source and target vertex tags and carrier type.

        Output Format:
            <source_vertex>                                    -><target_vertex>                                      Carrier:<carrier>

        Behavior:
            - Iterates through all edges in the graph.
            - Retrieves the 'Tag' attribute of the source and target vertices.
            - Retrieves the 'carrier' attribute of the edge.
            - Prints each edge in a structured format for visual inspection.

        Attributes Accessed:
            - self.g.es: List of edges in the graph.
            - self.g.vs: List of vertices in the graph.
            - edge.source: Index of the source vertex.
            - edge.target: Index of the target vertex.
            - edge["carrier"]: Carrier type associated with the edge.
            - self.g.vs[...]["Tag"]: Human-readable identifier for each vertex.

        Example Output:
            Pipe                                             ->VALVE_001                                           Carrier:Flow

        Use Case:
            Useful for debugging, auditing graph structure, or tracing semantic relationships.
        """
        for edge in self.g.es:
            source_vertex = self.g.vs[edge.source]["Tag"]
            target_vertex = "->"+self.g.vs[edge.target]["Tag"]
            carrier = "Carrier:"+edge["carrier"]
            print(f"{source_vertex:<40} {target_vertex:^60} {carrier:^80}")
    
    def apply_rule(self, source_type:str, target_type:str, rulenumber:int, strength=0.99, effect_factor=1, time_constant=0, rationale=None, apply_carrier=None, searchpath=None, shortest=False, findValves=False, inverse=False): 
        """
        Applies a semantic rule by adding directed edges between vertices of specified carrier types based on valid paths.

        Parameters:
            source_type (str): Carrier type of source vertices (e.g., "Temperature").
            target_type (str): Carrier type of target vertices (e.g., "State").
            rulenumber (int): Identifier for the rule being applied.
            strength (float): Edge weight representing influence strength (default: 0.99).
            effect_factor (float): Lambda factor indicating directionality or polarity of influence (default: 1).
            time_constant (float): Tau value representing temporal delay (default: 0).
            rationale (str, optional): Textual justification or annotation for the rule.
            apply_carrier (str, optional): Carrier label to assign to the new edge (overrides default logic).
            searchpath (List[List[int]], optional): Precomputed list of valid vertex paths.
            shortest (bool): If True, applies only the shortest valid path per source-target pair.
            findValves (bool): If True, searches for intermediate "State"-tagged children as actors.
            inverse (bool): If True, reverses the direction of the edge (target → source).

        Returns:
            int: Number of edges successfully added to the graph.

        Behavior:
            - Iterates over all source vertices with the specified source_type.
            - Filters valid paths from source to target vertices using searchpath.
            - Optionally restricts to shortest paths or includes valve-related actors.
            - Determines the carrier type for each edge using source, target, or applied carrier.
            - Adds directed edges with specified attributes and optional actor metadata.
            - Supports inverse edge direction for backward propagation modeling.

        Example:
            To model control influence from Temperature to Valve State:
            apply_rule("Temperature", "State", rulenumber=31, apply_carrier="control", findValves=True)

        Notes:
            - Paths must begin at the source vertex and end at the target vertex.
            - Intermediate vertices are scanned for "State"-tagged children if findValves is enabled.
            - Edge attributes include carrier, weight, lambda_factor, tau, rationale, and actors.
        """
        def determine_carrier(source_vertex, target_vertex, applied_carrier):
            # Determine the carrier for the new edge
            if source_vertex["Carrier"] == target_vertex["Carrier"]:
                return source_vertex["Carrier"]
            elif applied_carrier is not None:
                return applied_carrier
            else:
                return "mixed"

        if searchpath is None:
            searchpath=self.vertexpaths

        num_edges_added = 0
        for source_vertex in self.g.vs.select(Carrier_eq=source_type):
            written = False
            valid_paths = [path for path in searchpath if path[0] == source_vertex.index]
            if not valid_paths:  # If no valid paths for this source, move to next source vertex
                continue
            if shortest:
                # Sort paths by length, shortest first
                valid_paths.sort(key=len)
            for target_vertex in self.g.vs.select(Carrier_eq=target_type):
                for path in valid_paths:
                    if path[-1] == target_vertex.index:
                        
                        # Check for "State" named children if findValves is True
                        actors = []
                        if findValves:
                            for vertex_id in path[1:-1]: # Looping over the intermediate vertices of the path
                                vertex = self.g.vs[vertex_id]
                                for child_id in self.g.successors(vertex):
                                    edge = self.g.es[self.g.get_eid(vertex_id, child_id)]
                                    if edge["carrier"] == "haschild":
                                        child = self.g.vs[child_id]
                                        if "State" in child["Tag"]:
                                            actors.append(child["Tag"])
                        else:
                            actors = None
                        
                        current_carrier = determine_carrier(source_vertex, target_vertex, apply_carrier)
                        if inverse:
                            tempedge = self.g.add_edge(path[-1], path[0], carrier=current_carrier, weight=strength, rulenumber=rulenumber, tau=time_constant, lambda_factor=effect_factor, rationale=rationale, Actors=actors)
                        else:
                            tempedge = self.g.add_edge(path[0], path[-1], carrier=current_carrier, weight=strength, rulenumber=rulenumber, tau=time_constant, lambda_factor=effect_factor, rationale=rationale, Actors=actors)
                        num_edges_added += 1
                        written = True
                        break  # Stop after finding the first (or shortest) valid path to a target_vertex
                if shortest and written:  # If an edge has been added, stop for this source_vertex
                    break   
        return num_edges_added

    def _vertex_has_parent(self, vertex, parent_tag):
        """
        Determines whether a given vertex has a parent with a specified tag by exploring semantic paths.

        Parameters:
            vertex (igraph.Vertex): The vertex to evaluate.
            parent_tag (str): The tag of the parent vertex to search for.

        Returns:
            bool: True if the vertex has a parent with the specified tag, False otherwise.

        Behavior:
            - Uses `explore_edges` to trace a path from the vertex through 'hasparent' and 'Product' carriers.
            - Extracts the last vertex in the path and checks whether its tag matches the given parent_tag.
            - Assumes that the last path returned by `explore_edges` is the most relevant or deepest.

        Notes:
            - Carrier limits are set to [3, 1] for 'hasparent' and 'Product', respectively.
            - This method is useful for validating hierarchical relationships in CAEX-like graphs.

        Example:
            If vertex "Sensor_001" is connected via 'hasparent' and 'Product' to "Module_A",
            then _vertex_has_parent(vertex, "Module_A") returns True.
        """
        # Check if the vertex has a certain parent
        parent_vertex_ids = list(self.explore_edges(vertex.index, ['hasparent', 'Product'], [3, 1]))[-1]
        return parent_tag in self.g.vs(parent_vertex_ids)["Tag"][0]

    def plot_graph3D(self):
        """
        Visualizes the graph structure in a 3D space using Plotly with a Kamada-Kawai layout.

        Features:
            - Filters out vertices with 'Carrier' == 'hierarch' to exclude structural nodes.
            - Computes 2D Kamada-Kawai layout and embeds it in 3D space with z=0.
            - Draws directed edges with arrowheads using `Scatter3d` and `Cone` objects.
            - Colors nodes based on their 'color' attribute and labels them with their 'Tag'.
            - Adds camera positioning and annotations for enhanced presentation.

        Returns:
            None. Displays an interactive 3D Plotly figure in the browser.

        Visualization Details:
            - Nodes are rendered as colored spheres with hoverable tags.
            - Edges are rendered as gray lines with directional cones to indicate flow.
            - Scene axes and tick labels are hidden for a cleaner layout.
            - Title and annotation describe the layout and context.

        Example Use Case:
            Useful for inspecting causal or control graphs, especially in systems modeled with Arroyo principles.

        Notes:
            - Requires Plotly (`import plotly.graph_objects as go`) and a valid igraph layout.
            - Assumes each vertex has 'Tag' and 'color' attributes, and each edge has 'weight' and 'carrier'.
        """
        vertices = [(v, v.index) for v in self.g.vs if v['Carrier'] != 'hierarch']
        labels = [v['Tag'] for v, _ in vertices]
        colors = [v['color'] for v, _ in vertices]
        N = len(labels)
        E = [e.tuple for e in self.g.es.select(weight_gt=0)]
        layt = self.g.layout('kk') 

        Xn = [layt[index][0] for _, index in vertices]
        Yn = [layt[index][1] for _, index in vertices]
        Xe = []
        Ye = []
        vertex_indices = [index for _, index in vertices]

        for e in E:
            if e[0] in vertex_indices and e[1] in vertex_indices:
                Xe += [layt[e[0]][0], layt[e[1]][0], None]
                Ye += [layt[e[0]][1], layt[e[1]][1], None]

        trace1 = go.Scatter3d(x=Xe,
                            y=Ye,
                            z=[0]*len(Xe),
                            mode='lines',
                            line=dict(color='rgb(210,210,210)', width=3),
                            hoverinfo='none'
                            )

        arrowheads = []
        for i in range(0, len(Xe), 3):
            arrowhead = go.Cone(x=[Xe[i+1]], y=[Ye[i+1]], z=[0],
                                u=[Xe[i+1]-Xe[i]], v=[Ye[i+1]-Ye[i]], w=[0],
                                sizemode='scaled',
                                sizeref=0.2,
                                anchor='tail',
                                showscale=False,
                                colorscale=[[0, 'rgb(210,210,210)'], [1, 'rgb(210,210,210)']],
                                hoverinfo='none'
                            )
            arrowheads.append(arrowhead)

        trace2 = go.Scatter3d(x=Xn,
                            y=Yn,
                            z=[0]*len(Xn),
                            mode='markers',
                            name='ntw',
                            marker=dict(symbol='circle',
                                        size=20,
                                        color=colors,
                                        line=dict(color='rgb(50,50,50)', width=2.5)
                                        ),
                            text=labels,
                            hoverinfo='text'
                            )

        layout = go.Layout(showlegend=False,
                        scene=dict(camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.5)),  
                                    xaxis=dict(showticklabels=False), 
                                    yaxis=dict(showticklabels=False),  
                                    zaxis=dict(showticklabels=False)), 
                        width=2400,
                        height=1400,
                        autosize=False,
                        title="ACDG",
                        font=dict(size=18),
                        annotations=[
                            dict(
                                showarrow=False,
                                text='This igraph.Graph has the Kamada-Kawai layout',
                                xref='paper',
                                yref='paper',
                                x=0,
                                y=-0.1,
                                xanchor='left',
                                yanchor='bottom',
                                font=dict(size=34)
                            )
                        ]
                        )

        data = [trace1, trace2] + arrowheads
        fig = go.Figure(data=data, layout=layout)
        fig.show()



    def plot_graph(self):
        """
        Visualizes the graph in 2D using Plotly with directional arrows, node labels, and interactive edge tooltips.

        Features:
            - Filters out vertices with 'Carrier' == 'hierarch' to exclude structural nodes.
            - Assigns descriptive names to edges including source/target tags, weight, and lambda factor.
            - Uses Kamada-Kawai layout for node positioning, scaled for better spacing.
            - Renders edges as gray lines with directional arrow annotations.
            - Computes edge midpoints and overlays invisible markers for hoverable edge metadata.
            - Adds hoverable edge labels at both the start and end of each edge for improved readability.
            - Colors nodes based on their 'color' attribute and labels them with their 'Tag'.

        Returns:
            None. Displays an interactive 2D Plotly figure in the browser.

        Visualization Details:
            - Nodes are rendered as colored circular markers with hoverable labels.
            - Edges are rendered as gray lines with directional arrows indicating flow.
            - Edge metadata (e.g., weight, lambda) is shown via hover text at three locations:
                - Midpoint of the edge
                - Near the source node
                - Near the target node
            - Scene axes and tick labels are hidden for a clean layout.
            - Title and font settings are customized for presentation.

        Example Use Case:
            Ideal for inspecting causal, control, or dependency graphs, especially in systems modeled with  principles.

        Notes:
            - Requires Plotly (`import plotly.graph_objects as go`) and a valid igraph layout.
            - Assumes each vertex has 'Tag' and 'color' attributes.
            - Assumes each edge has 'weight', 'lambda_factor', and 'carrier' attributes.
            - Hover markers are invisible but provide rich interactivity for edge inspection.
        """
        def assign_edge_names():
            for edge in self.g.es:
                source_tag=self.g.vs[edge.source]["Tag"]
                target_tag=self.g.vs[edge.target]["Tag"]
                edge_weight =edge['weight']
                edge_lambda=edge['lambda_factor']
                edge["name"]=f"{source_tag}=>{target_tag}\n({edge_weight},{edge_lambda})"
                
        assign_edge_names()
        vertices = [(v, v.index) for v in self.g.vs if v['Carrier'] != 'hierarch']
        labels = [v['Tag'] for v, _ in vertices]
        colors = [v['color'] for v, _ in vertices]
        N = len(labels)
        E = [e.tuple for e in self.g.es.select(weight_gt=0)]
        scale_factor=5
        layt = self.g.layout('kk')
        layt=[(x*scale_factor,y*scale_factor) for x,y in layt]
        Xn = [layt[index][0] for _, index in vertices]
        Yn = [layt[index][1] for _, index in vertices]
        Xe = []
        Ye = []

        vertex_indices = [index for _, index in vertices]
        for e in E:
            if e[0] in vertex_indices and e[1] in vertex_indices:
                Xe += [layt[e[0]][0], layt[e[1]][0], None]
                Ye += [layt[e[0]][1], layt[e[1]][1], None]

        # Compute midpoints for each edge to display hover text properly
        Xm = [(Xe[i] + Xe[i+1]) / 2 for i in range(0, len(Xe)-2, 3)]
        Ym = [(Ye[i] + Ye[i+1]) / 2 for i in range(0, len(Ye)-2, 3)]
        edge_names = [self.g.es[i]["name"] for i in range(len(self.g.es)) if self.g.es[i].tuple in E]

        # Add invisible markers for better hover effect on edges
        trace1 = go.Scatter(
            x=Xe,
            y=Ye,
            mode='lines',
            line=dict(color='rgb(210,210,210)', width=3),
            hoverinfo="none"  # Hover on lines might not work well
        )
        # Compute hover points near edge start
        X_start_hover = []
        Y_start_hover = []
        edge_start_names = []

        offset = 0.25  # small offset to move away from the node

        for i in range(0, len(Xe), 3):
            x0, y0 = Xe[i], Ye[i]
            x1, y1 = Xe[i+1], Ye[i+1]

            dx = x1 - x0
            dy = y1 - y0
            length = (dx**2 + dy**2)**0.5

            # Offset point near source
            x_hover = x0 + dx / length * offset
            y_hover = y0 + dy / length * offset

            X_start_hover.append(x_hover)
            Y_start_hover.append(y_hover)

            edge_index = i // 3
            edge_start_names.append(edge_names[edge_index])

        # Add invisible hover markers near edge start
        trace_edge_start_hover = go.Scatter(
            x=X_start_hover,
            y=Y_start_hover,
            mode='markers',
            marker=dict(size=6, color='rgba(160,160,160,1)'),  # Gray dot
            hoverinfo="text",
            text=edge_start_names
        )
        # Compute hover points near edge end
        X_end_hover = []
        Y_end_hover = []
        edge_end_names = []

        for i in range(0, len(Xe), 3):
            x0, y0 = Xe[i], Ye[i]
            x1, y1 = Xe[i+1], Ye[i+1]

            dx = x1 - x0
            dy = y1 - y0
            length = (dx**2 + dy**2)**0.5

            # Offset point near target
            x_hover = x1 - dx / length * offset
            y_hover = y1 - dy / length * offset

            X_end_hover.append(x_hover)
            Y_end_hover.append(y_hover)

            edge_index = i // 3
            edge_end_names.append(edge_names[edge_index])

        # Add invisible hover markers near edge end
        trace_edge_end_hover = go.Scatter(
            x=X_end_hover,
            y=Y_end_hover,
            mode='markers',
            marker=dict(size=6, color='rgba(160,160,160,1)'),  # Gray dot
            hoverinfo="text",
            text=edge_end_names
        )


        # NEW: Add a separate scatter trace for edge midpoints
        trace_edges = go.Scatter(
            x=Xm,
            y=Ym,
            mode='markers',
            marker=dict(size=6, color='rgba(0,0,0,0)'),  # Invisible markers for hover
            hoverinfo="text",
            text=edge_names
        )
        trace2 = go.Scatter(x=Xn,
                            y=Yn,
                            mode='markers',
                            name='ntw',
                            marker=dict(symbol='circle-dot',
                                        size=20,
                                        color=colors,
                                        line=dict(color='rgb(50,50,50)', width=2.5)
                                        ),
                            text=labels,
                            hoverinfo='text'
                            )
        


        arrow_annotations = []
        arrow_scale = 0.03  # this is the constant distance from the circle's edge where the arrow will start/end
        for i in range(0, len(Xe), 3):
            edge_start = (Xe[i], Ye[i])
            edge_end = (Xe[i+1], Ye[i+1])

            dx = edge_end[0] - edge_start[0]
            dy = edge_end[1] - edge_start[1]
            length = (dx*dx + dy*dy)**0.5

            x_end_adj = edge_end[0] - dx/length*arrow_scale
            y_end_adj = edge_end[1] - dy/length*arrow_scale
            x_start_adj = edge_start[0] + dx/length*arrow_scale
            y_start_adj = edge_start[1] + dy/length*arrow_scale

            arrow_annotations.append(
                go.layout.Annotation(
                    x=x_end_adj,
                    y=y_end_adj,
                    ax=x_start_adj,
                    ay=y_start_adj,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.25,
                    arrowwidth=1,
                    arrowcolor='#636363',
                )
            )

        layout = go.Layout(title="ACDG",
                        font=dict(size=18),
                        showlegend=False,
                        autosize=False,
                        width=2400,
                        height=1400,
                        xaxis=go.layout.XAxis(showline=False, zeroline=False, showgrid=False, showticklabels=False),
                        yaxis=go.layout.YAxis(showline=False, zeroline=False, showgrid=False, showticklabels=False),
                        hovermode='closest',
                        annotations=arrow_annotations
                        )

        data = [trace1,trace_edges,trace_edge_start_hover,trace_edge_end_hover, trace2]
        fig = go.Figure(data=data, layout=layout)
        fig.show()


######################################################################################################################################################################################################################################################################################
 
    def generate_distancematrix(self): 
        """
        Generates a distance matrix for all currently active alarms.

        This method creates a matrix that contains the distance between all active alarms
        in the graph. The distance is calculated using a modified Dijkstra algorithm. In 
        order to use addition and subtraction the logarithm of the edge weights is summed 
        along the path between two alarms, and then transformed back from logspace using 
        an exponential function and then transformed back from logspace using an 
        exponential function.

        Distance are calculated from row to column, meaning that the distance_matrix[i, j] contains distance from alarm i to alarm j.

        Returns:
            np.ndarray: A 2D numpy array representing the distance matrix between active alarms.
        """
        active_alarm_list = [v for v in self.g.vs if v["AS"] != 0]
        # Generates a matrix that contains the distance between all active alarms
        distance_matrix = np.zeros((len(active_alarm_list), len(active_alarm_list)))
        for i in range(len(active_alarm_list)):
            for j in range(0, len(active_alarm_list)):#j in range(i + 1, len(active_alarm_list)):
                if i!=j:
                    distance = self.modified_dijkstra(active_alarm_list[i].index, active_alarm_list[j].index)
                    distance_matrix[i, j] = np.exp(-distance)
                else:
                    distance_matrix[i, j] = 0
        self.current_distancematrix = distance_matrix
        self.current_activealarmlist = active_alarm_list
        return distance_matrix
    
    
    def generate_distance(self, i, active_alarm_list, distance_matrix):
        """
        Computes and updates the distance matrix for a given alarm index using a modified Dijkstra algorithm.

        For each alarm `j` in the `active_alarm_list`, this method calculates the shortest path distance
        from alarm `i` to alarm `j` using `self.modified_dijkstra`. The resulting distance is transformed
        using an exponential decay function and stored in `distance_matrix[i, j]`. Diagonal entries are set to 0.

        Args:
            i (int): Index of the source alarm in `active_alarm_list`.
            active_alarm_list (List[Alarm]): List of active alarms, each with an `index` attribute.
            distance_matrix (np.ndarray): A 2D NumPy array to store the computed distances.

        Returns:
            None: The function updates `distance_matrix` in place.
        """
        for j in range(len(active_alarm_list)):
            if i != j:
                distance = self.modified_dijkstra(active_alarm_list[i].index, active_alarm_list[j].index)
                distance_matrix[i, j] = np.exp(-distance)
            else:
                distance_matrix[i, j] = 0

    def generate_distancematrix_multithreaded(self):
        """
        Generates a pairwise distance matrix for active alarms using multithreading.

        This method identifies all active alarms (where vertex attribute "AS" ≠ 0) from the graph `self.g`,
        and computes a distance matrix using a modified Dijkstra algorithm. The computation is parallelized
        using a thread pool to improve performance. Each distance is transformed using an exponential decay
        and stored in the matrix.

        The resulting matrix and active alarm list are stored in:
            - `self.current_distancematrix`
            - `self.current_activealarmlist`

        Returns:
            np.ndarray: A 2D NumPy array representing the computed distance matrix.
        """
        active_alarm_list = [v for v in self.g.vs if v["AS"] != 0]
        distance_matrix = np.zeros((len(active_alarm_list), len(active_alarm_list)))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            chunksize = 3  # Set the chunk size to 3
            executor.map(self.generate_distance, range(len(active_alarm_list)), 
                         [active_alarm_list]*len(active_alarm_list), 
                         [distance_matrix]*len(active_alarm_list), 
                         chunksize=chunksize)

        self.current_distancematrix = distance_matrix
        self.current_activealarmlist = active_alarm_list
        return distance_matrix


    def evaluate_distance_matrix(self, distance_limit, update_distance=False):
        """
        Evaluates the current distance matrix to identify clusters of alarms based on a distance threshold.

        This method groups alarms into clusters where each alarm can reach another within the specified
        `distance_limit`. Optionally, it updates the distance matrix before evaluation. Each cluster is
        analyzed to identify a representative alarm (RC) based on a custom influence metric, and clusters
        are labeled accordingly.

        Args:
            distance_limit (float): Threshold for determining whether two alarms are connected.
            update_distance (bool): If True, regenerates the distance matrix before evaluation.
                                    Uses multithreading if the number of active alarms is ≥ 9.

        Returns:
            Tuple[List[List[int]], List[List[str]], List[List[float]]]:
                - `clusters`: List of clusters, each containing indices of alarms.
                - `named_alarm_clusters`: List of clusters with alarm names, marking RC alarms with "RC:".
                - `division_values_clusters`: List of influence values for each alarm in each cluster.
        """
        # Finds clusters of alarms that can be reached from each other by a distance smaller than the distance_limit
        if update_distance:
            if len(self.current_activealarmlist) >= 9:
                self.generate_distancematrix_multithreaded()
            else:
                self.generate_distancematrix()
        # Check if distance_matrix exists and is a square matrix
        distance_matrix = self.current_distancematrix
        if distance_matrix is None or distance_matrix.shape[0] != distance_matrix.shape[1]:
            return "Invalid distance matrix"
        
        clusters = []
        
        for i in range(distance_matrix.shape[0]):            
            for j in range(distance_matrix.shape[1]):
                if distance_matrix[i, j] >= distance_limit:
                    clusters_i = [cluster for cluster in clusters if i in cluster]
                    clusters_j = [cluster for cluster in clusters if j in cluster]
                        
                    if not clusters_i and not clusters_j:
                        # Neither i nor j are in any cluster, create a new one
                        clusters.append([i, j])
                    elif clusters_i and not clusters_j:
                        # i is in existing clusters but j is not, add j to i's clusters
                        for cluster in clusters_i:
                            cluster.append(j)
                    elif clusters_j and not clusters_i:
                        # j is in existing clusters but i is not, add i to j's clusters
                        for cluster in clusters_j:
                            cluster.append(i)
                    else:
                        if clusters_i[0] is clusters_j[0]:
                            continue
                        # Both i and j are in existing clusters, merge them
                        merged_cluster = list(set(clusters_i[0] + clusters_j[0]))
                        clusters.remove(clusters_i[0])
                        clusters.remove(clusters_j[0])
                        clusters.append(merged_cluster)
                            
        # Sort each cluster by index
        for cluster in clusters:
            cluster.sort()
                       

        named_alarm_clusters = []
        division_values_clusters = []  # To store the division values for each cluster
        
        for cluster in clusters:
            current_cluster_names = []
            
            max_value = -1
            rc_alarm = None
            
            division_values = []  # To store the division values for the current cluster
            
            for i in cluster:
                sum_from_i = sum(distance_matrix[i, :])
                sum_to_i = sum(distance_matrix[:, i])
                
                divisor = sum_to_i if sum_to_i != 0 else 0.1
                value = sum_from_i / divisor
                
                division_values.append(value)  # Store the value
                
                if value > max_value:
                    max_value = value
                    rc_alarm = i

            division_values_clusters.append(division_values)  # Store the values for the current cluster
            
            for i in cluster:
                alarm_name = self.current_activealarmlist[i]["Tag"]
                if i == rc_alarm:
                    alarm_name = "RC:" + alarm_name
                current_cluster_names.append(alarm_name)
            
            if current_cluster_names:
                named_alarm_clusters.append(current_cluster_names)
                
        # After forming the clusters, add each "isolated" element as its own cluster
        all_elements = set(range(distance_matrix.shape[0]))
        clustered_elements = set(x for sublist in clusters for x in sublist)
        isolated_elements = all_elements - clustered_elements

        for elem in isolated_elements:
            clusters.append([elem])
            named_alarm_clusters.append([self.current_activealarmlist[elem]["Tag"]])
            division_values_clusters.append([0])  # or whatever value you want to assign to isolated elements
                    
        return clusters, named_alarm_clusters, division_values_clusters


    
    def evaluate_distance_matrix_working(self, distance_limit, update_distance=False):
        """
        Evaluates the current distance matrix to identify clusters of alarms based on a distance threshold.

        This method groups alarms into clusters where each alarm can reach another with a distance
        greater than or equal to the specified `distance_limit`. Optionally, it updates the distance matrix
        before evaluation. Clusters are merged dynamically based on connectivity, and alarm names are
        extracted for each cluster.

        Args:
            distance_limit (float): Threshold for determining whether two alarms are connected.
            update_distance (bool): If True, regenerates the distance matrix before evaluation.

        Returns:
            Tuple[List[List[int]], List[List[str]]]:
                - `clusters`: List of clusters, each containing indices of alarms.
                - `named_alarm_clusters`: List of clusters with alarm names from `self.current_activealarmlist`.
        """
        # Finds clusters of alarms that can be reached from each other by a distance smaller than the distance_limit
        if update_distance:
            self.generate_distancematrix()
        # Check if distance_matrix exists and is a square matrix
        distance_matrix = self.current_distancematrix
        if distance_matrix is None or distance_matrix.shape[0] != distance_matrix.shape[1]:
            return "Invalid distance matrix"
        
        clusters = []
        
        for i in range(distance_matrix.shape[0]):
            current_clusters_i_belongs_to = [cluster for cluster in clusters if i in cluster]
            
            if not current_clusters_i_belongs_to:
                new_cluster = [i]
                clusters.append(new_cluster)
                current_clusters_i_belongs_to = [new_cluster]
            
            for j in range(distance_matrix.shape[1]):
                if distance_matrix[i, j] >= distance_limit:
                    current_clusters_j_belongs_to = [cluster for cluster in clusters if j in cluster]
                    
                    if not current_clusters_j_belongs_to:
                        # Simply add j to the same clusters as i
                        for cluster in current_clusters_i_belongs_to:
                            cluster.append(j)
                    else:
                        # Merge all clusters that i and j belong to
                        merged_cluster = list(set.union(*map(set, current_clusters_i_belongs_to + current_clusters_j_belongs_to)))
                        
                        # Remove old clusters safely by creating a new list and filtering out the old ones
                        clusters = [cluster for cluster in clusters if cluster not in (current_clusters_i_belongs_to + current_clusters_j_belongs_to)]
                        
                        # Add the merged cluster
                        clusters.append(merged_cluster)
        named_alarm_clusters = []
        for cluster in clusters:
            current_cluster_names = []
            for i in cluster:
                current_cluster_names.append(self.current_activealarmlist[i]["Tag"])
            if current_cluster_names:
                named_alarm_clusters.append(current_cluster_names)
                
        return clusters, named_alarm_clusters


    def set_alarmstates(self, alarmstates_dict):
        """
        Updates the alarm state ("AS") attribute for vertices in the graph based on a provided dictionary.

        Each key in `alarmstates_dict` corresponds to a vertex's "Tag" attribute, and its value sets the
        corresponding "AS" (alarm state) attribute in the graph.

        Args:
            alarmstates_dict (Dict[str, Any]): A dictionary mapping vertex tags to alarm state values.

        Returns:
            None: The graph is updated in place.
        """
        for key in alarmstates_dict:
            self.g.vs(self.g.vs["Tag"].index(key))["AS"] = alarmstates_dict[key]

######################################################################################################################################################################################################################################################################################


