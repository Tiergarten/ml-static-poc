from static import *


def strings_cnt(file_path):
    printable_cnt = 0

    with open(file_path) as fd:
        while True:
            c = fd.read(1)
            if not c:
                break
            if c in string.printable:
                printable_cnt += 1

    return printable_cnt


def get_section_size(pe, section_name):
    for section in pe.sections:
        if section.Name.rstrip('\0') == section_name:
            return section.SizeOfRawData


# TODO: Use YARA rules to enrich features: [ is_packed, contains_base64, ... ]
def get_feature_header():
    return ["strings_cnt", "import_cnt", "text_sz", "rdata_sz", "data_sz", "rsrc_sz"]


def get_features(file_path):
    pe = pefile.PE(file_path, fast_load=False)

    data = [strings_cnt(file_path), len(pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0]
    for section in [".text", ".rdata", ".data", ".rsrc"]:
        sz = get_section_size(pe, section)
        data.append(sz if sz != None else 0)

    return data


def get_dir_features_rdd(dir_path, label, sc, overwrite=False):
    data = []
    fromPickle = False

    install_dir = os.path.dirname(os.path.realpath(__file__))
    pq_dir = install_dir + "/" + re.sub("[^A-Za-z0-9]", "", dir_path) + "-analysed"

    if os.path.isdir(pq_dir) and not overwrite:
        fromPickle = True
        ret = sc.pickleFile(pq_dir)
    else:
        for f in listdir(dir_path):
            fq = join(dir_path, f)
            if isfile(fq):
                features = get_features(fq)
                logging.debug("%s -> %s", fq, str(features))
                data.append(LabeledPoint(label, get_features(fq)))

            df = get_df(data)
            df.rdd.saveAsPickleFile(pq_dir)
            ret = df.rdd

    logging.info("Loaded %d samples from %s (Pre-processed: %s)", len(ret.collect()), dir_path, fromPickle)
    return ret