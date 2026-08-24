import pandas as pd
import random
import os


class solution_holder:
    def __init__(self, id, decisions, objective, rank, fidelity=None):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank
        self.fidelity = fidelity


class file_data:
    def __init__(self, name, all_set, training_set, test_set, 
                independent_set, features, dict_search,
                fidelity_columns=None, fidelity_values=None):
        self.name = name
        self.all_set = all_set
        self.training_set = training_set
        self.test_set = test_set
        self.independent_set = independent_set
        self.features = features
        self.dict_search = dict_search
        self.fidelity_columns = fidelity_columns
        self.fidelity_values = fidelity_values


def sort_and_deduplicate_columns(pdcontent, indepcolumns):
    tmp_sortindepcolumns = []
    for col in indepcolumns:
        tmp_sortindepcolumns.append(sorted(list(set(pdcontent[col]))))
    return tmp_sortindepcolumns


def create_ranks(sortpdcontent, depcolumns):
    ranks = {}
    if len(depcolumns) > 0:
        for i, item in enumerate(sorted(set(sortpdcontent[depcolumns[-1]].tolist()))):
            ranks[item] = i
    return ranks


def create_content(sortpdcontent, indepcolumns, depcolumns, ranks, fidelity_columns=None):
    content = []
    for c in range(len(sortpdcontent)):
        fidelity = None
        if fidelity_columns:
            fidelity = sortpdcontent.iloc[c][fidelity_columns].tolist()
        content.append(solution_holder(
            c,
            sortpdcontent.iloc[c][indepcolumns].tolist(),
            sortpdcontent.iloc[c][depcolumns].tolist(),
            ranks[sortpdcontent.iloc[c][depcolumns].tolist()[-1]],
            fidelity
        ))
    return content


def get_data(filename, initial_size=10,seed=1,verbose=False):
    try:
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        if os.path.isabs(filename):
            full_path = filename
        else:
            cleaned_filename = filename.lstrip('/\\')
            full_path = os.path.join(project_root, cleaned_filename)
        full_path = os.path.normpath(full_path)
        pdcontent = pd.read_csv(full_path)

        columns_to_invert = [col for col in pdcontent.columns if col.startswith('+$<')]
        for col in columns_to_invert:
            pdcontent[col] = -pdcontent[col]

        fidelity_columns = [col for col in pdcontent.columns if col.startswith('F$')]
        indepcolumns = [col for col in pdcontent.columns if "$<" not in col and col not in fidelity_columns]
        depcolumns = [col for col in pdcontent.columns if "$<" in col]

        if len(indepcolumns) == 0 or len(depcolumns) == 0:
            return None

        tmp_sortindepcolumns = sort_and_deduplicate_columns(pdcontent, indepcolumns)
        fidelity_values = {}
        if fidelity_columns:
            for col in fidelity_columns:
                unique_fidelities = sorted(pdcontent[col].dropna().unique())
                fidelity_values[col] = unique_fidelities

        sortpdcontent = pdcontent.sort_values(by=depcolumns[-1])
        ranks = create_ranks(sortpdcontent, depcolumns)
        content = create_content(sortpdcontent, indepcolumns, depcolumns, ranks, fidelity_columns)

        dict_search = dict(zip([tuple(i.decision) for i in content], [i.objective[-1] for i in content]))

        random.seed(seed)
        shuffled_content = random.sample(content, len(content))
        training_set = shuffled_content[:initial_size]
        test_set = shuffled_content[initial_size:]

        file = file_data(
            filename, content, training_set, test_set,
            tmp_sortindepcolumns, indepcolumns, dict_search,
            fidelity_columns, fidelity_values
        )
        return file
    except FileNotFoundError:
        pass
    except IndexError:
        pass
    except Exception:
        pass


def load_features (features_fileName):
    header = features_fileName.features
    features = [t.decision for t in features_fileName.all_set]
    target = [t.objective[-1] for t in features_fileName.all_set]
    return header , features , target
