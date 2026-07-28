# Class name to label index

EVENT_DICTIONARY_action_class = {"Tackling":0,"Standing tackling":1,"High leg":2,"Holding":3,"Pushing":4,
                        "Elbowing":5, "Challenge":6, "Dive":7, "Dont know":8} # 动作类别

INVERSE_EVENT_DICTIONARY_action_class = {0:"Tackling", 1:"Standing tackling", 2:"High leg", 3:"Holding", 4:"Pushing",
                        5:"Elbowing", 6:"Challenge", 7:"Dive", 8:"Dont know"} # 逆向类别


EVENT_DICTIONARY_offence_severity_class = {"No offence":0,"Offence + No card":1,"Offence + Yellow card":2,"Offence + Red card":3} # 犯规严重程度类别

INVERSE_EVENT_DICTIONARY_offence_severity_class = {0:"No offence", 1:"Offence + No card", 2:"Offence + Yellow card", 3:"Offence + Red card"} # 逆向犯规严重程度类别


EVENT_DICTIONARY_offence_class = {"Offence":0,"Between":1,"No Offence":2, "No offence":2} # 犯规类别

INVERSE_EVENT_DICTIONARY_offence_class = {0:"Offence", 1:"Between", 2:"No Offence"} # 逆向犯规类别


EVENT_DICTIONARY_severity_class = {"1.0":0,"2.0":1,"3.0":2,"4.0":3,"5.0":4} # 严重程度类别

INVERSE_EVENT_DICTIONARY_severity_class = {0:"No card", 1:"Borderline No/Yellow", 2:"Yellow card", 3:"Borderline Yellow/Red", 4:"Red card"} # 逆向严重程度类别


EVENT_DICTIONARY_bodypart_class = {"Upper body":0,"Under body":1} # 身体部位类别

INVERSE_EVENT_DICTIONARY_bodypart_class = {0:"Upper body", 1:"Under body"} # 逆向身体部位类别



EVENT_DICTIONARY = {'action_class':EVENT_DICTIONARY_action_class, 'offence_class': EVENT_DICTIONARY_offence_class, 
            'severity_class': EVENT_DICTIONARY_severity_class, 'bodypart_class': EVENT_DICTIONARY_bodypart_class, 'offence_severity_class': EVENT_DICTIONARY_offence_severity_class}
INVERSE_EVENT_DICTIONARY = {'action_class':INVERSE_EVENT_DICTIONARY_action_class, 'offence_class': INVERSE_EVENT_DICTIONARY_offence_class, 
            'severity_class': INVERSE_EVENT_DICTIONARY_severity_class, 'bodypart_class': INVERSE_EVENT_DICTIONARY_bodypart_class, 'offence_severity_class': INVERSE_EVENT_DICTIONARY_offence_severity_class}
