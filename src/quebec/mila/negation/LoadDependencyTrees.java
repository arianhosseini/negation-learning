package quebec.mila.negation;

import java.io.BufferedWriter;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.semgrex.SemgrexMatcher;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;
import edu.stanford.nlp.semgraph.semgrex.ssurgeon.AddDep;
import edu.stanford.nlp.semgraph.semgrex.ssurgeon.AddEdge;
import edu.stanford.nlp.semgraph.semgrex.ssurgeon.RemoveNamedEdge;
import edu.stanford.nlp.semgraph.semgrex.ssurgeon.SsurgeonEdit;
import edu.stanford.nlp.semgraph.semgrex.ssurgeon.SsurgeonPattern;
import edu.stanford.nlp.trees.EnglishGrammaticalRelations;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.ud.CoNLLUDocumentReader;
import edu.stanford.nlp.util.Pair;

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
	
	public static void negateSentence(SemanticGraph sg) throws Exception {
		
		System.out.println("Start = "+sg.toCompactString());
		SemgrexPattern semgrexPattern = SemgrexPattern.compile("{}=A >nsubj=E {}=B");
		
		// This part is for understanding. Don't need this one.
		SemgrexMatcher matcher = semgrexPattern.matcher(sg);
	    while (matcher.find()) {
	    	System.out.println(String.format("A: %s", matcher.getNode("A")));
	    	System.out.println(String.format("B: %s", matcher.getNode("B")));
	    }
	    
	    // Match and edit the tree 
	    SsurgeonPattern pattern = new SsurgeonPattern(semgrexPattern);
	    
		// Attach not
		IndexedWord notNode = new IndexedWord();
		notNode.set(CoreAnnotations.TextAnnotation.class, "not");
		notNode.set(CoreAnnotations.LemmaAnnotation.class, "not");
		notNode.set(CoreAnnotations.OriginalTextAnnotation.class, "not");
		notNode.set(CoreAnnotations.PartOfSpeechAnnotation.class, "PART");
		SsurgeonEdit addNot = new AddDep("A", EnglishGrammaticalRelations.ADVERBIAL_MODIFIER, notNode);
		pattern.addEdit(addNot);

		Collection<SemanticGraph> newSgs = pattern.execute(sg);
		for (SemanticGraph newSg : newSgs) {
			System.out.println("Modified = "+newSg.toCompactString());
			System.out.println("\n\n");
			System.out.println(printSemanticGraph(newSg));
		}
	}

	public static void readTreesFromFile(String inputFile) throws Exception {
		Iterator<Pair<SemanticGraph, SemanticGraph>> sgIterator;
	    CoNLLUDocumentReader reader = new CoNLLUDocumentReader();
	    sgIterator = reader.getIterator(IOUtils.readerFromString(inputFile));
	    while (sgIterator.hasNext()) {
	     System.out.println("================================");
	      SemanticGraph sg = sgIterator.next().first;
	      System.out.println(printSemanticGraph(sg));
	      
	      System.out.println("\n\n");
	      negateSentence(sg);
	      System.out.println("\n\n");
	    }
	}

	public static void main(String[] args) throws Exception {
		LoadDependencyTrees.readTreesFromFile("data/mnli.examples.txt");
	}
}
