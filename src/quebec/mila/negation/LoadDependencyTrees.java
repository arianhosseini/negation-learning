package quebec.mila.negation;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.List;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.ud.CoNLLUDocumentReader;

public class LoadDependencyTrees {

	public static String printSemanticGraph(SemanticGraph sg) {
		StringBuilder sb = new StringBuilder();
		List<IndexedWord> tokens = sg.vertexListSorted();

		for (IndexedWord token : tokens) {
			IndexedWord parentToken = sg.getParent(token);
			int parentIndex = parentToken == null ? 0 : parentToken.index();
			String parentWord = parentToken == null ? "root" : parentToken.word();
			String label = parentToken == null ? GrammaticalRelation.ROOT.toString()
					: sg.getEdge(parentToken, token).getRelation().toString();

			sb.append(String.format("%d\t%s\t_\t%s\t_\t_\t%d\t%s\t_\t_%n", token.index(), token.word(),
					parentWord, parentIndex, label));
		}
		sb.append("\n");
		return sb.toString();
	}

	public static void readTreesFromFile(String inputFile) throws IOException {
		// PrintStream out = new PrintStream(system
	    // BufferedWriter fout =
	    //    new BufferedWriter(new OutputStreamWriter(out, StandardCharsets.UTF_8));

	    Iterator<SemanticGraph> sgIterator;
	    CoNLLUDocumentReader reader = new CoNLLUDocumentReader();
	    sgIterator = reader.getIterator(IOUtils.readerFromString(inputFile));
	    while (sgIterator.hasNext()) {
	      SemanticGraph sg = sgIterator.next();
	      // tag(sg);
	      // fout.write(printSemanticGraph(sg));
	      System.out.println(printSemanticGraph(sg));
	    }

	    // fout.close();
	    // out.close();
	}

	public static void main(String[] args) throws IOException {
		LoadDependencyTrees.readTreesFromFile("data/conll_wiki1m.simple.txt");
	}
}
