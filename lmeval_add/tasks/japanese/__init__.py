# Only one class from each task_file is required to call for 
# initializing and adding into ALL_TASKS, all the tasks available in the file.

from .jsquad import JSQuAD
from .jaqket_v1 import JAQKETV1
from .jaqket_v2 import JAQKETV2
from .jaquad import JaQuAD
from .jcsqa import JCommonsenseQA
from .jnli import JNLIWithFintanPrompt
from .marc_ja import MARCJaWithFintanPrompt
from .xlsum import XLSumJa
from .mgsm import MGSM
from .wikilingua import Wikilingua
