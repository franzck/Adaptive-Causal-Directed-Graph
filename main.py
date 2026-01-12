from ACDG import ACDG_Class

######################################################################################################################################################################################################################################################################################

######################### MAIN #########################
if __name__ == "__main__":
    aml_file = r"C:/path/to/your/file.aml"
    acdg = ACDG_Class()
    print("Generating graph from AML file")
    acdg.generate_graphfromAML(aml_file)
    print("Graph generated")
    print("Generating distance matrix")
    acdg.generate_distancematrix()
    print("Distance matrix generated")
    acdg.plot_graph()
    