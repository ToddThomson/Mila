# Framing (WTP/2)

A frame is at most 16384 bytes on the wire, everything included.

Every frame starts with the 12-byte base header: version (1), flags (1), stream id (4), sequence (4) and
length (2). When the X flag is set, an 8-byte extended header follows the base header (`flags.md`).

What follows the headers is the frame body. Without encryption the body is the application payload as given.
With encryption the body is laid out as `security.md` describes.

A sender that has more application data than one frame can carry splits it across frames; it never sends an
oversized frame.
