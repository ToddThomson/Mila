# Newsletter layout is broken

From the editor, after this week's issue went out:

1. The justified paragraphs look lopsided -- the wide gaps all bunch up at the right-hand end of each line. The
   docstring says the leftmost gaps should get the extra space. One of the tests catches this.
2. Any paragraph pasted from the CMS comes out mangled. The CMS puts two spaces after full stops and hard line
   breaks inside paragraphs, e.g. `"Thanks for reading.  See you\nnext week"` -- words end up glued across the
   line break and there are stray double gaps.

Both need fixing before the next issue.
