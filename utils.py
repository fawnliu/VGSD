from os.path import join, isfile
from shutil import copytree, rmtree, ignore_patterns

def backup_code(src_dir, dest_dir):
    # ignore any files but files with '.py' extension
    def get_ignored(dir, filenames):
        to_exclude = ['datasets', 'models']
        ret = []
        if dir in to_exclude:
            ret.append(dir)
        for filename in filenames:
            if join(dir, filename) in to_exclude:
                ret.append(filename)
            elif isfile(join(dir, filename)) and not filename.endswith(".py") and  not filename.endswith(".sh"):
                ret.append(filename)
        # print(ret)
        return ret
    # ignore_func = lambda d, files: [f for f in files if isfile(join(d, f)) and not f.endswith('.py') and not f.endswith('.sh')]
    rmtree(dest_dir, ignore_errors=True)
    copytree(src_dir, dest_dir, ignore=ignore_patterns("datasets*", "models*", ".git*", "*.pth", "results*",
                                                       "logs*", "slurm_logs", "__pycache__"))
