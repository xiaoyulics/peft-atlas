"""Convert the CFF-flavoured OpenType faces to TrueType so matplotlib embeds them as clean Type 42 fonts."""
import os, sys
from fontTools.ttLib import TTFont, newTable
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

def otf_to_ttf(src, dst, max_err=1.0):
    font = TTFont(src)
    glyph_order = font.getGlyphOrder()
    font['loca'] = newTable('loca'); font['glyf'] = glyf = newTable('glyf')
    glyf.glyphOrder = glyph_order; glyf.glyphs = {}
    gs = font.getGlyphSet()
    for name in glyph_order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, max_err, reverse_direction=True))
        glyf[name] = pen.glyph()
    font['maxp'] = maxp = newTable('maxp'); maxp.tableVersion = 0x00010000
    for attr in ('maxZones', 'maxTwilightPoints', 'maxStorage', 'maxFunctionDefs', 'maxInstructionDefs', 'maxStackElements', 'maxSizeOfInstructions', 'maxComponentElements'):
        setattr(maxp, attr, 0)
    maxp.maxZones = 1
    post = font['post']; post.formatType = 2.0; post.extraNames = []; post.mapping = {}; post.glyphOrder = glyph_order
    font['head'].glyphDataFormat = 0
    del font['CFF ']
    if 'VORG' in font: del font['VORG']
    font.sfntVersion = '\x00\x01\x00\x00'
    font.save(dst)

if __name__ == '__main__':
    from style import TEXMF, FONT_FILES
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
    for f in FONT_FILES:
        src = os.path.join(TEXMF, f); dst = os.path.join(out, os.path.basename(f).replace('.otf', '.ttf'))
        otf_to_ttf(src, dst); print('wrote', dst)
