def sql_pattern_formatting(patterns):
    """
    Formats a string or list for an SQL LIKE statement.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    for p in patterns:
        if not isinstance(patterns[p], str):
            raise TypeError(f'"Input "{p}" to parameter `patterns` must be of type `str`"')
    formatted_pattern = ["'% " + p + "%'" for p in patterns]
    return formatted_pattern

def sql_like_statement(patterns, matching_column = "company_name", escape_expression = "@"):
    """
    Create a SQL LIKE statement for pattern-search in a matching variable. 
    
    Parameters:
    ----------
    patterns : list
        A list of patterns to search in the matching variable.
    matching_variable: str
        A string specifiying the JPOD column to be searched for the pattern. Defaults to 'company_name'
    escape_expression: str
        A string indicating that wildcard characters in SQL ('%', '_') are matched with their literal values. 
        Defaults to '@'.
    Returns:
    --------
    str:
        A string in a SQL LIKE Statement format.
    """
    formatted_patterns = sql_pattern_formatting(patterns)
    if len(formatted_patterns) > 1:
        match_string = " OR %s LIKE " % matching_column
        like_statement = match_string.join(formatted_patterns)
        like_statement = matching_column + " LIKE " + like_statement + " ESCAPE '%s'" %escape_expression
    else:
        like_statement = str(matching_column) + " LIKE '" + patterns[0] + "' ESCAPE '%s'" %escape_expression
    return like_statement

def dict_items_to_list(input_dict):
    out_list = []
    for v in input_dict.values():
        out_list += v
    return out_list

p = 1