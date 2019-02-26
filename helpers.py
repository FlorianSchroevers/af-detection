import sys
import sympy
import unicodedata
from global_params import cfg

def progress_bar_serious(message, part, total):
    message = message + ': '
    counter = str(part+1) + '/' + str(total) + ' '
    message = message + '\n' + counter
    if part + 1 == total:
        print(message + '['  + '=' * (int(cfg.t_width * 0.75)) +  ']  ', end = '\n')
    else:
        p  = int(((part+1)/total) * int(cfg.t_width * 0.75))
        up = int(cfg.t_width * 0.75) - p
        print(message + '[' + '=' * (p-1) + '>' + '-' * (up) + ']  ', end = '\n')

        sys.stdout.write("\033[F") #back to previous line
        sys.stdout.write("\033[K")
        sys.stdout.write("\033[F") #back to previous line
        sys.stdout.write("\033[K")

def progress_bar_childish(message, part, total):
    message = message + ': '
    counter = str(part+1) + '/' + str(total) + ' '
    message = message + '\n' + counter
    if part + 1 == total:
        sympy.pprint(message + '8' + '=' * (int(cfg.t_width * 0.75)) + 'D' + '~~~' + ' ')
    else:
        p  = int(((part+1)/total) * int(cfg.t_width * 0.75))
        up = int(cfg.t_width * 0.75) - p
        sympy.pprint(message + '8' + '=' * (p-1) + 'D' + ' ' * (up) + unicodedata.lookup("GREEK SMALL LETTER EPSILON") + ' ')

        sys.stdout.write("\033[F") #back to previous line
        sys.stdout.write("\033[K")
        sys.stdout.write("\033[F") #back to previous line
        sys.stdout.write("\033[K")

pbd = {
    "childish":progress_bar_childish,
    "serious":progress_bar_serious
}
progress_bar = pbd[cfg.progress_bar_style]
